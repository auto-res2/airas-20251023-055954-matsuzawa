import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
import numpy as np
import random
import os
from pathlib import Path
import wandb
import optuna
from copy import deepcopy

# ---------------------------------------------------------------------------
# Robust intra-package imports
# ---------------------------------------------------------------------------
try:
    from .preprocess import get_dataloaders  # type: ignore
    from .model import (
        build_model,
        edcp_score,
        logistic_sigmoid_score,
    )  # type: ignore
except ImportError:  # pragma: no cover
    from preprocess import get_dataloaders  # type: ignore
    from model import (
        build_model,
        edcp_score,
        logistic_sigmoid_score,
    )  # type: ignore

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = preds.eq(targets).sum().item()
        return correct / targets.size(0)

# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------

def train_once(cfg: DictConfig, trial: optuna.Trial | None = None):
    if trial is not None:
        # Inject Optuna-suggested hyper-parameters
        for hp_name, space in cfg.run.optuna.search_space.items():
            if space["type"] == "loguniform":
                sampled = trial.suggest_float(hp_name, space["low"], space["high"], log=True)
            elif space["type"] == "uniform":
                sampled = trial.suggest_float(hp_name, space["low"], space["high"], log=False)
            elif space["type"] == "categorical":
                sampled = trial.suggest_categorical(hp_name, space["choices"])
            elif space["type"] == "int":
                sampled = trial.suggest_int(hp_name, space["low"], space["high"], step=1)
            else:
                raise ValueError(f"Unsupported search-space type: {space['type']}")
            target = cfg.run
            parts = hp_name.split(".")
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = sampled

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(cfg.run.dataset, cfg.run.training, mode=cfg.mode)
    model = build_model(cfg.run.model).to(device)

    criterion = nn.CrossEntropyLoss()

    opt_cfg = cfg.run.training.optimizer
    if opt_cfg.type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg.learning_rate,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay,
            nesterov=True,
        )
    elif opt_cfg.type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.learning_rate, weight_decay=opt_cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer {opt_cfg.type}")

    sched_cfg = cfg.run.training.scheduler
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.run.training.epochs, eta_min=sched_cfg.min_lr)
        if sched_cfg.type.lower() == "cosine"
        else None
    )

    best_val_acc = 0.0
    val_curve: list[float] = []

    for epoch in range(cfg.run.training.epochs):
        # ---------------- training -----------------
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_seen = 0
        for batch_idx, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += logits.argmax(1).eq(y).sum().item()
            train_seen += bs

            if cfg.mode == "trial" and batch_idx >= 1:
                break

        train_loss = train_loss_sum / max(train_seen, 1)
        train_acc = train_correct / max(train_seen, 1)

        # ---------------- validation --------------
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_seen = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(loaders["val"]):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                bs = y.size(0)
                val_loss_sum += loss.item() * bs
                val_correct += logits.argmax(1).eq(y).sum().item()
                val_seen += bs

                if cfg.mode == "trial" and batch_idx >= 1:
                    break

        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = val_correct / max(val_seen, 1)
        val_curve.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)

        if scheduler is not None:
            scheduler.step()

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    # -------- curve compression --------
    cc = cfg.run.curve_compression
    if cc.name.lower() == "edcp":
        compressed_score = edcp_score(val_curve, gamma=cc.gamma)
    elif cc.name.lower() == "logistic_sigmoid":
        compressed_score = logistic_sigmoid_score(val_curve, m0=cc.m0, g0=cc.g0)
    else:
        raise ValueError(f"Unknown curve compression {cc.name}")

    metrics = {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_curve[-1],
        "compressed_score": compressed_score,
    }

    if wandb.run is not None:
        for k, v in metrics.items():
            wandb.summary[k] = v

    return metrics

# ---------------------------------------------------------------------------
# Optuna objective wrapper
# ---------------------------------------------------------------------------

def _objective(cfg: DictConfig):
    def _inner(trial: optuna.Trial):
        tmp_cfg = deepcopy(cfg)
        out = train_once(tmp_cfg, trial)
        return out["compressed_score"]

    return _inner

# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    set_seed(42)

    # ---------------- WandB ----------------
    if cfg.wandb.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None
    else:
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB URL: {wandb_run.get_url()}")

    # ------------- HPO via Optuna ----------
    if cfg.run.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.run.optuna.direction)
        study.optimize(_objective(cfg), n_trials=cfg.run.optuna.n_trials)
        for k, v in study.best_trial.params.items():
            target = cfg.run
            parts = k.split(".")
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = v

    # ------------- Final training ----------
    final_metrics = train_once(cfg)

    # ------------- Save artefacts ----------
    ckpt_dir = Path(hydra.utils.to_absolute_path(cfg.results_dir)) / cfg.run.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"metrics": final_metrics}, ckpt_dir / "model_final.pth")
    print(f"Saved metrics to {ckpt_dir / 'model_final.pth'}")

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()