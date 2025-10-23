import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import shlex
import subprocess
import yaml

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode not in {"trial", "full"}:
        raise ValueError("mode must be 'trial' or 'full'")

    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.run.optuna.n_trials = 0
        cfg.run.training.epochs = 1
        cfg.run.training.batch_size = min(cfg.run.training.batch_size, 16)
    else:
        cfg.wandb.mode = "online"

    results_dir = Path(cfg.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = results_dir / "config.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)
    print(f"Saved final config to {cfg_path}")

    overrides = [
        f"run={cfg.run.run_id}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
        f"wandb.mode={cfg.wandb.mode}",
        f"run.optuna.n_trials={cfg.run.optuna.n_trials}",
        f"run.training.epochs={cfg.run.training.epochs}",
        f"run.training.batch_size={cfg.run.training.batch_size}",
    ]

    cmd = ["python", "-u", "-m", "src.train"] + overrides
    print("Launching training subprocess:\n" + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()