"""Part 3 (Notebook Step 4 & Step 5): Chronological split, split EDA, and checkpoints.

Splits each user's ratings chronologically into Train (80%), Validation (10%), and
Test (10%) splits, validates zero temporal leakage, analyzes cold-start sparsity
categories on the test set, and saves/verifies all checkpoint files under
``processed_v1/``.
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    ensure_dir,
    load_json,
    load_multimodal_inputs,
    save_json,
    save_or_show_figure,
)


def chronological_per_user_split(
    multimodal_ratings: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions chronologically per user while guaranteeing >=1 row per split."""
    sorted_ratings = multimodal_ratings.sort_values(["userId", "timestamp"]).reset_index(
        drop=True
    )

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    val_cum_ratio = train_ratio + val_ratio
    for _, group in sorted_ratings.groupby("userId", sort=False):
        n = len(group)
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * val_cum_ratio))
        if val_end >= n:
            val_end = n - 1

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_ratings = pd.concat(train_parts, ignore_index=True)
    val_ratings = pd.concat(val_parts, ignore_index=True)
    test_ratings = pd.concat(test_parts, ignore_index=True)
    return train_ratings, val_ratings, test_ratings


def verify_split_integrity(
    multimodal_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
) -> dict[str, int]:
    """Verify split shapes, user coverage across splits, and chronological order."""
    for name, df in [
        ("Train", train_ratings),
        ("Validation", val_ratings),
        ("Test", test_ratings),
    ]:
        print(f"\n{name}")
        print("Rows:", len(df))
        print("Users:", df["userId"].nunique())
        print("Movies:", df["movieId"].nunique())
        print("Mean rating:", round(df["rating"].mean(), 3))
        print("Median rating:", df["rating"].median())

    train_users = set(train_ratings["userId"])
    val_users = set(val_ratings["userId"])
    test_users = set(test_ratings["userId"])
    users_in_all = len(train_users & val_users & test_users)

    latest_train = train_ratings.groupby("userId")["timestamp"].max()
    earliest_val = val_ratings.groupby("userId")["timestamp"].min()
    latest_val = val_ratings.groupby("userId")["timestamp"].max()
    earliest_test = test_ratings.groupby("userId")["timestamp"].min()

    train_val_violations = int((latest_train > earliest_val).sum())
    val_test_violations = int((latest_val > earliest_test).sum())

    print("\nUsers in all three splits:", users_in_all)
    print("Total users:", multimodal_ratings["userId"].nunique())
    print("Actual Train -> Validation violations:", train_val_violations)
    print("Actual Validation -> Test violations:", val_test_violations)

    return {
        "users_in_all_splits": users_in_all,
        "train_val_violations": train_val_violations,
        "val_test_violations": val_test_violations,
    }


def analyze_split_sparsity(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Analyze unseen movie coverage and test-movie training sparsity buckets."""
    train_movies = set(train_ratings["movieId"])
    val_movies = set(val_ratings["movieId"])
    test_movies = set(test_ratings["movieId"])

    val_unseen = val_movies - train_movies
    test_unseen = test_movies - train_movies

    print("\nValidation movies unseen in training:", len(val_unseen))
    print("Test movies unseen in training:", len(test_unseen))
    print(
        "Validation unseen percentage:",
        round(len(val_unseen) / len(val_movies) * 100, 2),
        "%",
    )
    print(
        "Test unseen percentage:",
        round(len(test_unseen) / len(test_movies) * 100, 2),
        "%",
    )

    train_movie_counts = train_ratings.groupby("movieId").size().rename("train_rating_count")
    test_movie_analysis = (
        test_ratings[["movieId"]]
        .drop_duplicates()
        .merge(train_movie_counts, on="movieId", how="left")
    )
    test_movie_analysis["train_rating_count"] = test_movie_analysis[
        "train_rating_count"
    ].fillna(0)

    print("\nTest movie training-count describe:")
    print(test_movie_analysis["train_rating_count"].describe())
    print(
        "Test movies with zero training ratings:",
        int((test_movie_analysis["train_rating_count"] == 0).sum()),
    )
    print(
        "Test movies with <= 5 training ratings:",
        int((test_movie_analysis["train_rating_count"] <= 5).sum()),
    )
    print(
        "Test movies with <= 10 training ratings:",
        int((test_movie_analysis["train_rating_count"] <= 10).sum()),
    )
    print(
        "Test movies with <= 20 training ratings:",
        int((test_movie_analysis["train_rating_count"] <= 20).sum()),
    )

    counts = test_movie_analysis["train_rating_count"]
    sparse_summary = pd.DataFrame(
        {
            "Condition": [
                "Zero training ratings",
                "1–5 training ratings",
                "6–10 training ratings",
                "11–20 training ratings",
                "More than 20 training ratings",
            ],
            "Movie Count": [
                int((counts == 0).sum()),
                int(((counts >= 1) & (counts <= 5)).sum()),
                int(((counts >= 6) & (counts <= 10)).sum()),
                int(((counts >= 11) & (counts <= 20)).sum()),
                int((counts > 20).sum()),
            ],
        }
    )
    sparse_summary["Percentage"] = (
        sparse_summary["Movie Count"] / len(test_movie_analysis) * 100
    )
    print("\nSparse-feedback summary:\n", sparse_summary)

    coverage_stats = {
        "val_unseen_movies": float(len(val_unseen)),
        "test_unseen_movies": float(len(test_unseen)),
    }
    return test_movie_analysis, sparse_summary, coverage_stats


def plot_split_distributions(
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    test_movie_analysis: pd.DataFrame,
    plots_dir: Path | None = None,
    show_plots: bool = False,
) -> None:
    """Plot split rating histograms and test-movie training rating count distributions."""
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.hist(train_ratings["rating"], bins=10, alpha=0.7, label="Train")
    ax1.hist(val_ratings["rating"], bins=10, alpha=0.7, label="Validation")
    ax1.hist(test_ratings["rating"], bins=10, alpha=0.7, label="Test")
    ax1.set_title("Rating Distribution Across Dataset Splits")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Number of Ratings")
    ax1.legend()
    save_or_show_figure(
        fig1,
        plots_dir / "step04_split_rating_distribution.png" if plots_dir else None,
        show_plots=show_plots,
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(test_movie_analysis["train_rating_count"], bins=50)
    ax2.set_title("Training Ratings Available for Test Movies")
    ax2.set_xlabel("Number of Training Ratings per Movie")
    ax2.set_ylabel("Number of Test Movies")
    ax2.set_xscale("log")
    save_or_show_figure(
        fig2,
        plots_dir / "step04_test_movie_train_count_dist.png" if plots_dir else None,
        show_plots=show_plots,
    )


def save_checkpoint_files(
    checkpoint_dir: Path,
    train_ratings: pd.DataFrame,
    val_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    multimodal_movies: pd.DataFrame,
    test_movie_analysis: pd.DataFrame,
    sparse_summary: pd.DataFrame,
    multimodal_ratings: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """Save Step 5 checkpoint parquets, sparsity summaries, and dataset metadata."""
    ensure_dir(checkpoint_dir)

    train_ratings.to_parquet(checkpoint_dir / "train_ratings.parquet", index=False)
    val_ratings.to_parquet(checkpoint_dir / "val_ratings.parquet", index=False)
    test_ratings.to_parquet(checkpoint_dir / "test_ratings.parquet", index=False)

    multimodal_movies[["movieId", "title", "genres"]].to_parquet(
        checkpoint_dir / "multimodal_movies.parquet", index=False
    )
    test_movie_analysis.to_parquet(
        checkpoint_dir / "test_movie_sparsity_analysis.parquet", index=False
    )
    sparse_summary.to_csv(checkpoint_dir / "sparse_feedback_summary.csv", index=False)

    source_stats_path = checkpoint_dir / "source_dataset_stats.json"
    source_stats = load_json(source_stats_path) if source_stats_path.exists() else {}

    clip_dim = source_stats.get("clip_dimension", config.clip_dim)
    audio_dim = source_stats.get("audio_dimension", config.audio_dim)
    if (checkpoint_dir / "clip_embeddings.npy").exists():
        clip_arr = np.load(checkpoint_dir / "clip_embeddings.npy", mmap_mode="r")
        clip_movies_count = int(clip_arr.shape[0])
        clip_dim = int(clip_arr.shape[1])
    else:
        clip_movies_count = int(source_stats.get("clip_movies", 0))

    if (checkpoint_dir / "vggish_embeddings.npy").exists():
        audio_arr = np.load(checkpoint_dir / "vggish_embeddings.npy", mmap_mode="r")
        audio_movies_count = int(audio_arr.shape[0])
        audio_dim = int(audio_arr.shape[1])
    else:
        audio_movies_count = int(source_stats.get("audio_movies", 0))

    movies_with_ratings = int(multimodal_ratings["movieId"].nunique())
    metadata = {
        "project": (
            "Leveraging Poster and Trailer-Audio Embeddings for "
            "Cold-Start Movie Recommendation"
        ),
        "version": "V1",
        "movielens": {
            "total_ratings_source": int(source_stats.get("total_ratings_source", 0)),
            "users": int(
                source_stats.get("source_users", multimodal_ratings["userId"].nunique())
            ),
            "movies": int(
                source_stats.get("source_movies", multimodal_movies["movieId"].nunique())
            ),
        },
        "multimodal": {
            "aligned_movies": int(len(multimodal_movies)),
            "movies_with_ratings": movies_with_ratings,
            "movies_without_ratings": int(len(multimodal_movies) - movies_with_ratings),
            "clip_movies": clip_movies_count,
            "audio_movies": audio_movies_count,
            "clip_dimension": clip_dim,
            "audio_dimension": audio_dim,
        },
        "interactions": {
            "multimodal_ratings": int(len(multimodal_ratings)),
            "users": int(multimodal_ratings["userId"].nunique()),
            "movies": movies_with_ratings,
        },
        "split": {
            "method": "per-user chronological split",
            "train_ratio": config.train_ratio,
            "validation_ratio": config.val_ratio,
            "test_ratio": round(1.0 - config.train_ratio - config.val_ratio, 4),
            "train_rows": int(len(train_ratings)),
            "validation_rows": int(len(val_ratings)),
            "test_rows": int(len(test_ratings)),
        },
    }
    save_json(metadata, checkpoint_dir / "dataset_metadata.json")
    print("Saved Step 5 checkpoints and dataset_metadata.json to:", checkpoint_dir)


def verify_checkpoints(checkpoint_dir: Path) -> dict[str, bool]:
    """Verify presence, file sizes, and reload shapes of Step 5 checkpoints."""
    expected_files = [
        "train_ratings.parquet",
        "val_ratings.parquet",
        "test_ratings.parquet",
        "multimodal_movies.parquet",
        "clip_embeddings.npy",
        "clip_movie_ids.npy",
        "vggish_embeddings.npy",
        "vggish_movie_ids.npy",
        "dataset_metadata.json",
        "test_movie_sparsity_analysis.parquet",
        "sparse_feedback_summary.csv",
    ]
    print("\n===== CHECKPOINT VERIFICATION =====")
    status: dict[str, bool] = {}
    for filename in expected_files:
        path = checkpoint_dir / filename
        exists = path.exists()
        status[filename] = exists
        if exists:
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"[OK]      {filename:<40} {size_mb:,.2f} MB")
        else:
            print(f"[MISSING] {filename}")

    if all(
        status[name]
        for name in (
            "train_ratings.parquet",
            "val_ratings.parquet",
            "test_ratings.parquet",
            "multimodal_movies.parquet",
        )
    ):
        print("Train:", pd.read_parquet(checkpoint_dir / "train_ratings.parquet").shape)
        print("Validation:", pd.read_parquet(checkpoint_dir / "val_ratings.parquet").shape)
        print("Test:", pd.read_parquet(checkpoint_dir / "test_ratings.parquet").shape)
        movies_df = pd.read_parquet(checkpoint_dir / "multimodal_movies.parquet")
        print("Multimodal movies:", movies_df.shape)
        print("Unique movie IDs:", movies_df["movieId"].nunique())
    return status


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    show_plots: bool = False,
    save_plots: bool = True,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Execute Stage 03 (Steps 4 & 5): split dataset, run split EDA, and save checkpoints."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    multimodal_ratings, multimodal_movies = load_multimodal_inputs(
        resolved_paths.checkpoint_dir
    )
    train_ratings, val_ratings, test_ratings = chronological_per_user_split(
        multimodal_ratings,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )
    integrity = verify_split_integrity(
        multimodal_ratings, train_ratings, val_ratings, test_ratings
    )
    test_movie_analysis, sparse_summary, coverage_stats = analyze_split_sparsity(
        train_ratings, val_ratings, test_ratings
    )

    plot_split_distributions(
        train_ratings=train_ratings,
        val_ratings=val_ratings,
        test_ratings=test_ratings,
        test_movie_analysis=test_movie_analysis,
        plots_dir=resolved_paths.plots_dir if save_plots else None,
        show_plots=show_plots,
    )

    if save_outputs:
        save_checkpoint_files(
            checkpoint_dir=resolved_paths.checkpoint_dir,
            train_ratings=train_ratings,
            val_ratings=val_ratings,
            test_ratings=test_ratings,
            multimodal_movies=multimodal_movies,
            test_movie_analysis=test_movie_analysis,
            sparse_summary=sparse_summary,
            multimodal_ratings=multimodal_ratings,
            config=cfg,
        )
        verify_checkpoints(resolved_paths.checkpoint_dir)

    return {
        "train_ratings": train_ratings,
        "val_ratings": val_ratings,
        "test_ratings": test_ratings,
        "test_movie_analysis": test_movie_analysis,
        "sparse_summary": sparse_summary,
        "integrity": integrity,
        "coverage_stats": coverage_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 03 (Steps 4 & 5): Chronological split, split EDA, and checkpoints."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib windows interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Skip saving split figures to processed_v1/plots/.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run split and verification without writing checkpoint files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        show_plots=args.show_plots,
        save_plots=not args.no_save_plots,
        save_outputs=not args.no_save,
    )
