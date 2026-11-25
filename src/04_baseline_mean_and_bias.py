"""Part 4 (Notebook Steps 6.1 & 6.2): Global Mean and User + Movie Bias baselines.

Evaluates two non-neural collaborative baselines strictly estimated from training
ratings (no validation/test leakage) and persists metrics for Part 8 comparison.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    compute_regression_metrics,
    load_checkpoint_splits,
    print_regression_metrics,
    save_model_metrics,
)


def evaluate_global_mean_baseline(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
) -> dict[str, object]:
    """Fit and evaluate Baseline 1 (Global Mean rating from training data)."""
    global_mean = float(train_ratings["rating"].mean())
    print(f"Global mean rating: {global_mean:.4f}")

    val_pred_mean = np.full(len(val_ratings), global_mean, dtype=np.float32)
    test_pred_mean = np.full(len(test_ratings), global_mean, dtype=np.float32)

    val_metrics = compute_regression_metrics(
        val_ratings["rating"], val_pred_mean, include_r2=True
    )
    test_metrics = compute_regression_metrics(
        test_ratings["rating"], test_pred_mean, include_r2=True
    )
    print_regression_metrics("Global Mean Baseline", val_metrics, test_metrics)

    return {
        "global_mean": global_mean,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def fit_user_movie_bias(
    train_ratings: pd.DataFrame,
) -> tuple[float, pd.Series, pd.Series]:
    """Estimate global mean, user biases, and movie biases from training ratings only."""
    global_mean = float(train_ratings["rating"].mean())
    user_stats = train_ratings.groupby("userId")["rating"].agg(["mean", "count"])
    user_bias = user_stats["mean"] - global_mean

    movie_stats = train_ratings.groupby("movieId")["rating"].agg(["mean", "count"])
    movie_bias = movie_stats["mean"] - global_mean

    print(f"\nGlobal mean: {global_mean:.4f}")
    print(f"Users with learned bias: {len(user_bias):,}")
    print(f"Movies with learned bias: {len(movie_bias):,}")
    return global_mean, user_bias, movie_bias


def predict_bias_model(
    df: pd.DataFrame,
    global_mean: float,
    user_bias: pd.Series,
    movie_bias: pd.Series,
    rating_min: float = 0.5,
    rating_max: float = 5.0,
) -> pd.Series:
    """Predict ``mu + b_u + b_i`` clipped to valid MovieLens rating bounds."""
    predictions = (
        global_mean
        + df["userId"].map(user_bias).fillna(0.0)
        + df["movieId"].map(movie_bias).fillna(0.0)
    )
    return predictions.clip(rating_min, rating_max)


def evaluate_bias_baseline(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """Fit and evaluate Baseline 2 (User + Movie Bias)."""
    cfg = config or PipelineConfig()
    global_mean, user_bias, movie_bias = fit_user_movie_bias(train_ratings)

    val_pred_bias = predict_bias_model(
        val_ratings, global_mean, user_bias, movie_bias, cfg.rating_min, cfg.rating_max
    )
    test_pred_bias = predict_bias_model(
        test_ratings, global_mean, user_bias, movie_bias, cfg.rating_min, cfg.rating_max
    )

    val_metrics = compute_regression_metrics(
        val_ratings["rating"], val_pred_bias, include_r2=True
    )
    test_metrics = compute_regression_metrics(
        test_ratings["rating"], test_pred_bias, include_r2=True
    )
    print_regression_metrics("\nUser + Movie Bias Baseline", val_metrics, test_metrics)

    return {
        "global_mean": global_mean,
        "num_users": int(len(user_bias)),
        "num_movies": int(len(movie_bias)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    save_metrics: bool = True,
) -> dict[str, dict[str, object]]:
    """Execute Stage 04 (Baselines 1 & 2) and save evaluation metrics."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    train_ratings, val_ratings, test_ratings = load_checkpoint_splits(
        resolved_paths.checkpoint_dir
    )
    global_mean_res = evaluate_global_mean_baseline(
        train_ratings, val_ratings, test_ratings
    )
    bias_res = evaluate_bias_baseline(
        train_ratings, val_ratings, test_ratings, config=cfg
    )

    if save_metrics:
        save_model_metrics(
            metrics_dir=resolved_paths.metrics_dir,
            filename="global_mean_metrics.json",
            model_name="Global Mean",
            val_mae=global_mean_res["val_metrics"]["mae"],
            val_rmse=global_mean_res["val_metrics"]["rmse"],
            test_mae=global_mean_res["test_metrics"]["mae"],
            test_rmse=global_mean_res["test_metrics"]["rmse"],
            evaluation="Overall",
            extra={
                "Notes": "Simple reference baseline",
                "Global Mean": global_mean_res["global_mean"],
                "Validation R2": global_mean_res["val_metrics"]["r2"],
                "Test R2": global_mean_res["test_metrics"]["r2"],
            },
        )
        save_model_metrics(
            metrics_dir=resolved_paths.metrics_dir,
            filename="bias_model_metrics.json",
            model_name="User + Movie Bias",
            val_mae=bias_res["val_metrics"]["mae"],
            val_rmse=bias_res["val_metrics"]["rmse"],
            test_mae=bias_res["test_metrics"]["mae"],
            test_rmse=bias_res["test_metrics"]["rmse"],
            evaluation="Overall",
            extra={
                "Notes": "Classical collaborative baseline",
                "Validation R2": bias_res["val_metrics"]["r2"],
                "Test R2": bias_res["test_metrics"]["r2"],
            },
        )
        print("\nSaved Baseline 1 & 2 metrics to:", resolved_paths.metrics_dir)

    return {
        "global_mean": global_mean_res,
        "bias_model": bias_res,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 04 (Steps 6.1 & 6.2): Evaluate Global Mean and User+Movie Bias baselines."
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Skip writing metric JSON files to processed_v1/metrics/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(save_metrics=not args.no_save_metrics)
