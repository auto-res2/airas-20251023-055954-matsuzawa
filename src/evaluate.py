import argparse
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
import wandb
from scipy import stats

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style():
    sns.set(style="whitegrid", font_scale=1.2)

def save_line(df: pd.DataFrame, x: str, y: str, title: str, path: Path):
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()

def save_bar(data: dict[str, float], title: str, path: Path):
    keys, vals = zip(*data.items())
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=list(keys), y=list(vals))
    for idx, v in enumerate(vals):
        ax.text(idx, v + 1e-3, f"{v:.4f}", ha="center", va="bottom")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()

def save_box(df: pd.DataFrame, y: str, hue: str, title: str, path: Path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, y=y, x=hue)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Comprehensive evaluation of multiple WandB runs")
    parser.add_argument("results_dir", type=str, help="Directory for output artefacts")
    parser.add_argument("run_ids", type=str, help="JSON list string of run IDs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    run_ids = json.loads(args.run_ids)

    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml expected at {cfg_path}")
    with cfg_path.open() as f:
        root_cfg = yaml.safe_load(f)
    entity = root_cfg["wandb"]["entity"]
    project = root_cfg["wandb"]["project"]

    api = wandb.Api()
    all_paths: list[Path] = []
    per_run_metrics: list[dict] = []

    _style()

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        history: pd.DataFrame = run.history()
        summary = dict(run.summary._json_dict)
        cfg = dict(run.config)

        run_dir = results_dir / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"history": history.to_dict(orient="list"), "summary": summary, "config": cfg}, indent=2)
        )
        all_paths.append(metrics_path)

        if "val_acc" in history.columns:
            lc_path = run_dir / f"{rid}_learning_curve.pdf"
            save_line(history, "_step", "val_acc", f"Validation Accuracy – {rid}", lc_path)
            all_paths.append(lc_path)
        if "train_acc" in history.columns:
            lc2_path = run_dir / f"{rid}_train_curve.pdf"
            save_line(history, "_step", "train_acc", f"Training Accuracy – {rid}", lc2_path)
            all_paths.append(lc2_path)

        if "val_confusion_matrix" in summary:
            cm = np.asarray(summary["val_confusion_matrix"])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix – {rid}")
            plt.tight_layout()
            cm_path = run_dir / f"{rid}_confusion_matrix.pdf"
            plt.savefig(cm_path, format="pdf")
            plt.close()
            all_paths.append(cm_path)

        per_run_metrics.append(
            {
                "run_id": rid,
                "method": cfg.get("method", "unknown"),
                "best_val_acc": summary.get("best_val_acc", np.nan),
                "final_val_acc": summary.get("final_val_acc", np.nan),
                "compressed_score": summary.get("compressed_score", np.nan),
            }
        )

    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(per_run_metrics)
    df.to_json(comp_dir / "aggregated_metrics.json", orient="records", indent=2)

    bar_data = df.groupby("method")["best_val_acc"].mean().to_dict()
    bar_path = comp_dir / "comparison_best_val_acc_bar_chart.pdf"
    save_bar(bar_data, "Mean Best Validation Accuracy (per method)", bar_path)
    all_paths.append(bar_path)

    box_path = comp_dir / "comparison_best_val_acc_box_plot.pdf"
    save_box(df, "best_val_acc", "method", "Best Validation Accuracy Distribution", box_path)
    all_paths.append(box_path)

    if "baseline" in df["method"].unique():
        baseline_vals = df[df["method"] == "baseline"]["best_val_acc"].values
        improvement_records = {}
        for m in df["method"].unique():
            if m == "baseline":
                continue
            other_vals = df[df["method"] == m]["best_val_acc"].values
            if len(other_vals) == 0:
                continue
            min_len = min(len(baseline_vals), len(other_vals))
            imp_rate = (np.mean(other_vals[:min_len]) - np.mean(baseline_vals[:min_len])) / np.mean(baseline_vals[:min_len])
            improvement_records[m] = imp_rate
        if improvement_records:
            imp_path = comp_dir / "improvement_rate_bar_chart.pdf"
            save_bar(improvement_records, "Relative Improvement over Baseline", imp_path)
            all_paths.append(imp_path)

    stats_results = {}
    if "baseline" in df["method"].unique():
        baseline_vals = df[df["method"] == "baseline"]["best_val_acc"].values
        for m in df["method"].unique():
            if m == "baseline":
                continue
            other_vals = df[df["method"] == m]["best_val_acc"].values
            if len(other_vals) < 2 or len(baseline_vals) < 2:
                warnings.warn(f"Insufficient samples for statistical test: baseline vs {m}")
                continue
            try:
                stat, p = stats.wilcoxon(baseline_vals[: len(other_vals)], other_vals)
                test_name = "wilcoxon"
            except ValueError:
                stat, p = stats.mannwhitneyu(baseline_vals, other_vals, alternative="two-sided")
                test_name = "mannwhitneyu"
            stats_results[m] = {"test": test_name, "statistic": float(stat), "p_value": float(p)}
    (comp_dir / "significance_tests.json").write_text(json.dumps(stats_results, indent=2))
    all_paths.append(comp_dir / "significance_tests.json")

    for p in all_paths:
        print(str(p))

if __name__ == "__main__":
    main()