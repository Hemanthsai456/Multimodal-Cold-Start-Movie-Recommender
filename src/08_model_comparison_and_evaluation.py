"""Part 8 (Notebook Step 9): Model comparison and evaluation reporting.

Aggregates saved metrics from Parts 4, 5, 6, and 7 (or optionally evaluates
saved checkpoints directly with ``--eval-saved-models``), computes relative
improvements over collaborative and cold-start baselines, exports
``final_results.csv``, and generates all comparison visualizations.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig, ProjectPaths
from src.utils import ensure_dir, import_step_module, load_json, save_or_show_figure

METRIC_FILENAMES: list[tuple[str, str]] = [
    ("global_mean_metrics.json", "04_baseline_mean_and_bias.py"),
    ("bias_model_metrics.json", "04_baseline_mean_and_bias.py"),
    ("ncf_metrics.json", "05_baseline_ncf.py"),
    ("multimodal_metrics.json", "06_multimodal_model.py"),
    ("coldstart_metrics.json", "07_coldstart_model.py"),
]


def load_comparison_metrics(
    metrics_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Load saved per-model metric JSONs produced by Parts 4–7."""
    missing = [
        (fname, producer)
        for fname, producer in METRIC_FILENAMES
        if not (metrics_dir / fname).exists()
    ]
    if missing:
        details = "; ".join(f"{fname} (from {prod})" for fname, prod in missing)
        raise FileNotFoundError(
            f"Missing model metric files in {metrics_dir}: {details}. "
            "Run Parts 4–7 first, or pass --eval-saved-models to evaluate saved checkpoints."
        )

    records_by_model: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for fname, _ in METRIC_FILENAMES:
        record = load_json(metrics_dir / fname)
        records_by_model[str(record["Model"])] = record
        rows.append(
            {
                "Model": record["Model"],
                "Validation MAE": (
                    np.nan
                    if record.get("Validation MAE") is None
                    else float(record["Validation MAE"])
                ),
                "Validation RMSE": (
                    np.nan
                    if record.get("Validation RMSE") is None
                    else float(record["Validation RMSE"])
                ),
                "Test MAE": float(record["Test MAE"]),
                "Test RMSE": float(record["Test RMSE"]),
                "Evaluation": record.get("Evaluation", "Overall"),
            }
        )

    final_results = pd.DataFrame(rows)
    return final_results, records_by_model


def evaluate_from_saved_checkpoints(
    paths: ProjectPaths,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Recompute all model metrics directly from checkpoint data and saved models."""
    import_step_module("04_baseline_mean_and_bias").run(
        paths=paths, config=config, save_metrics=True
    )
    import_step_module("05_baseline_ncf").run(
        paths=paths, config=config, evaluate_only=True, save_metrics=True
    )
    import_step_module("06_multimodal_model").run(
        paths=paths, config=config, evaluate_only=True, save_metrics=True
    )
    import_step_module("07_coldstart_model").run(
        paths=paths, config=config, evaluate_only=True, save_metrics=True
    )
    return load_comparison_metrics(paths.metrics_dir)


def compute_relative_improvements(
    records_by_model: dict[str, dict[str, object]],
) -> dict[str, float]:
    """Compute percentage improvements for Multimodal and Cold-Start models."""
    bias = records_by_model["User + Movie Bias"]
    mm = records_by_model["Multimodal"]
    cs = records_by_model["Cold-Start Multimodal"]

    bias_mae = float(bias["Test MAE"])
    bias_rmse = float(bias["Test RMSE"])
    mm_mae = float(mm["Test MAE"])
    mm_rmse = float(mm["Test RMSE"])

    mm_mae_improvement = ((bias_mae - mm_mae) / bias_mae) * 100.0
    mm_rmse_improvement = ((bias_rmse - mm_rmse) / bias_rmse) * 100.0

    cs_mae = float(cs["Test MAE"])
    cs_rmse = float(cs["Test RMSE"])
    cs_global_mae = float(cs["Global Mean MAE"])
    cs_global_rmse = float(cs["Global Mean RMSE"])

    cs_mae_improvement = ((cs_global_mae - cs_mae) / cs_global_mae) * 100.0
    cs_rmse_improvement = ((cs_global_rmse - cs_rmse) / cs_global_rmse) * 100.0

    print("\nMultimodal Improvement over User + Movie Bias")
    print(f"MAE improvement : {mm_mae_improvement:.2f}%")
    print(f"RMSE improvement: {mm_rmse_improvement:.2f}%")
    print("\nCold-Start Improvement over Global Mean (Unseen Subset)")
    print(f"MAE improvement : {cs_mae_improvement:.2f}%")
    print(f"RMSE improvement: {cs_rmse_improvement:.2f}%")

    return {
        "mm_mae_improvement": mm_mae_improvement,
        "mm_rmse_improvement": mm_rmse_improvement,
        "cs_mae": cs_mae,
        "cs_rmse": cs_rmse,
        "cs_global_mae": cs_global_mae,
        "cs_global_rmse": cs_global_rmse,
        "cs_mae_improvement": cs_mae_improvement,
        "cs_rmse_improvement": cs_rmse_improvement,
    }


def generate_comparison_plots(
    final_results: pd.DataFrame,
    improvements: dict[str, float],
    plots_dir: Path | None = None,
    show_plots: bool = False,
) -> None:
    """Generate the five Step 9 model evaluation and comparison charts."""
    # 1. Horizontal bar chart across all evaluated models
    models = final_results["Model"]
    mae = final_results["Test MAE"]
    rmse = final_results["Test RMSE"]
    y = np.arange(len(models))
    height = 0.32

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.barh(y - height / 2, mae, height, label="MAE")
    ax1.barh(y + height / 2, rmse, height, label="RMSE")
    ax1.set_yticks(y)
    ax1.set_yticklabels(models)
    ax1.invert_yaxis()
    ax1.set_xlabel("Error")
    ax1.set_title("Model Performance — Test Set")
    ax1.legend(loc="upper left", ncol=2)
    for spine in ("top", "right", "left"):
        ax1.spines[spine].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    fig1.tight_layout()
    save_or_show_figure(
        fig1,
        plots_dir / "step09_model_performance_barh.png" if plots_dir else None,
        show_plots=show_plots,
    )

    overall_results = final_results[final_results["Evaluation"] == "Overall"].copy()

    # 2. Overall Test MAE Comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    mae_bars = ax2.bar(overall_results["Model"], overall_results["Test MAE"])
    ax2.bar_label(mae_bars, fmt="%.4f", padding=3)
    ax2.set_ylabel("Test MAE")
    ax2.set_xlabel("Model")
    ax2.set_title("Test MAE Comparison")
    ax2.tick_params(axis="x", rotation=20)
    fig2.tight_layout()
    save_or_show_figure(
        fig2,
        plots_dir / "step09_test_mae_comparison.png" if plots_dir else None,
        show_plots=show_plots,
    )

    # 3. Overall Test RMSE Comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    rmse_bars = ax3.bar(overall_results["Model"], overall_results["Test RMSE"])
    ax3.bar_label(rmse_bars, fmt="%.4f", padding=3)
    ax3.set_ylabel("Test RMSE")
    ax3.set_xlabel("Model")
    ax3.set_title("Test RMSE Comparison")
    ax3.tick_params(axis="x", rotation=20)
    fig3.tight_layout()
    save_or_show_figure(
        fig3,
        plots_dir / "step09_test_rmse_comparison.png" if plots_dir else None,
        show_plots=show_plots,
    )

    # 4. Multimodal improvement over User + Movie Bias
    improvement_df = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE"],
            "Improvement (%)": [
                improvements["mm_mae_improvement"],
                improvements["mm_rmse_improvement"],
            ],
        }
    )
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    imp_bars = ax4.bar(improvement_df["Metric"], improvement_df["Improvement (%)"])
    ax4.bar_label(imp_bars, fmt="%.2f%%", padding=3)
    ax4.set_ylabel("Improvement (%)")
    ax4.set_xlabel("Metric")
    ax4.set_title("Multimodal Improvement over User + Movie Bias")
    fig4.tight_layout()
    save_or_show_figure(
        fig4,
        plots_dir / "step09_multimodal_improvement.png" if plots_dir else None,
        show_plots=show_plots,
    )

    # 5. Cold-Start subset evaluation comparison
    coldstart_comparison = pd.DataFrame(
        {
            "Model": ["Global Mean", "Cold-Start Multimodal"],
            "MAE": [improvements["cs_global_mae"], improvements["cs_mae"]],
            "RMSE": [improvements["cs_global_rmse"], improvements["cs_rmse"]],
        }
    )
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(coldstart_comparison))
    width = 0.35
    cs_mae_bars = ax5.bar(
        x - width / 2, coldstart_comparison["MAE"], width, label="MAE"
    )
    cs_rmse_bars = ax5.bar(
        x + width / 2, coldstart_comparison["RMSE"], width, label="RMSE"
    )
    ax5.bar_label(cs_mae_bars, fmt="%.4f", padding=3)
    ax5.bar_label(cs_rmse_bars, fmt="%.4f", padding=3)
    ax5.set_xticks(x)
    ax5.set_xticklabels(coldstart_comparison["Model"])
    ax5.set_ylabel("Error")
    ax5.set_title("Cold-Start Evaluation (Unseen Movies Subset)")
    ax5.legend()
    fig5.tight_layout()
    save_or_show_figure(
        fig5,
        plots_dir / "step09_coldstart_evaluation.png" if plots_dir else None,
        show_plots=show_plots,
    )


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    eval_saved_models: bool = False,
    show_plots: bool = False,
    save_plots: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Execute Stage 08 (Step 9): compare models, save ``final_results.csv``, and plot."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    if eval_saved_models:
        final_results, records_by_model = evaluate_from_saved_checkpoints(
            resolved_paths, cfg
        )
    else:
        final_results, records_by_model = load_comparison_metrics(
            resolved_paths.metrics_dir
        )

    print("\n===== FINAL MODEL COMPARISON =====")
    print(final_results.to_string(index=False))
    print(
        "\nNote: Cold-Start Multimodal is evaluated exclusively on movies with zero "
        "training interactions and should not be compared directly as an equivalent "
        "overall population metric."
    )

    improvements = compute_relative_improvements(records_by_model)

    if save_csv:
        ensure_dir(resolved_paths.metrics_dir)
        metrics_csv = resolved_paths.metrics_dir / "final_results.csv"
        final_results.to_csv(metrics_csv, index=False)

        ensure_dir(resolved_paths.recommendation_results_dir)
        results_csv = resolved_paths.recommendation_results_dir / "final_results.csv"
        final_results.to_csv(results_csv, index=False)
        print(f"\nSaved comparison CSV to: {metrics_csv} and {results_csv}")

    generate_comparison_plots(
        final_results=final_results,
        improvements=improvements,
        plots_dir=resolved_paths.plots_dir if save_plots else None,
        show_plots=show_plots,
    )
    return final_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 08 (Step 9): Aggregate model metrics, save comparison CSV, and plot."
    )
    parser.add_argument(
        "--eval-saved-models",
        action="store_true",
        help="Re-evaluate baselines and saved .keras checkpoints before comparing.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib windows interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Skip saving comparison plots to processed_v1/plots/.",
    )
    parser.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Skip saving final_results.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        eval_saved_models=args.eval_saved_models,
        show_plots=args.show_plots,
        save_plots=not args.no_save_plots,
        save_csv=not args.no_save_csv,
    )
