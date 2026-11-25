"""Part 7 (Notebook Step 8): Cold-Start Multimodal Model & Seen vs. Unseen Evaluation.

Analyzes seen vs. unseen test movies, trains a dedicated cold-start multimodal
recommender using an explicit ``UNK_MOVIE_INDEX`` slot and 20% random movie-ID
masking during training/validation, evaluates on true unseen test interactions
(zero training ratings), and breaks down error by test interaction count.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    build_multimodal_network,
    build_or_load_aligned_embeddings,
    build_or_load_id_mappings,
    compute_regression_metrics,
    create_multimodal_tf_dataset,
    ensure_dir,
    load_checkpoint_splits,
    prepare_multimodal_arrays,
    resolve_existing_file,
    save_model_metrics,
)


def analyze_seen_vs_unseen_test(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    multimodal_movie_index: dict[int, int],
) -> tuple[pd.DataFrame, set[int]]:
    """Split test interactions into seen vs. completely unseen training movies."""
    train_movie_set = set(train_ratings["movieId"].unique())
    test_ratings_analysis = test_ratings.copy()
    test_ratings_analysis["is_unseen_movie"] = ~test_ratings_analysis["movieId"].isin(
        train_movie_set
    )

    seen_mask = ~test_ratings_analysis["is_unseen_movie"]
    unseen_mask = test_ratings_analysis["is_unseen_movie"]

    print("Test interactions:", len(test_ratings_analysis))
    print("Seen interactions:", int(seen_mask.sum()))
    print("Unseen interactions:", int(unseen_mask.sum()))
    print(
        "\nSeen movies:",
        int(test_ratings_analysis.loc[seen_mask, "movieId"].nunique()),
    )
    print(
        "Unseen movies:",
        int(test_ratings_analysis.loc[unseen_mask, "movieId"].nunique()),
    )

    unseen_test_movies = set(
        test_ratings_analysis.loc[unseen_mask, "movieId"].unique()
    )
    unseen_with_both = unseen_test_movies & set(multimodal_movie_index.keys())
    print("Unseen movies with both modalities:", len(unseen_with_both))
    print(
        "Unseen test interactions with both modalities:",
        int(test_ratings_analysis["movieId"].isin(unseen_with_both).sum()),
    )
    return test_ratings_analysis, unseen_with_both


def build_coldstart_multimodal_recommender(
    num_users: int,
    num_movies: int,
    config: PipelineConfig | None = None,
) -> keras.Model:
    """Build Cold-Start Multimodal Recommender with an explicit UNK movie embedding."""
    return build_multimodal_network(
        num_users=num_users,
        movie_vocab_size=num_movies + 1,
        config=config,
    )


def apply_coldstart_masking(
    movie_indices: np.ndarray,
    unk_movie_index: int,
    mask_rate: float = 0.20,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly replace ``mask_rate`` of movie indices with ``unk_movie_index``."""
    if seed is None:
        rng = np.random.default_rng()
        mask = rng.random(len(movie_indices)) < mask_rate
    else:
        np.random.seed(seed)
        mask = np.random.rand(len(movie_indices)) < mask_rate

    masked = movie_indices.copy()
    masked[mask] = unk_movie_index
    return masked, mask


def assign_test_count_group(n: int) -> str:
    """Bucket unseen test movies by number of test interactions."""
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def evaluate_coldstart_subset(
    coldstart_model: keras.Model,
    test_ratings_analysis: pd.DataFrame,
    user_to_index: dict[int, int],
    multimodal_movie_index: dict[int, int],
    aligned_clip: np.ndarray,
    aligned_audio: np.ndarray,
    unk_movie_index: int,
    train_global_mean: float,
    config: PipelineConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate cold-start model on true unseen test movies and bucketed groups."""
    cfg = config or PipelineConfig()

    coldstart_test = test_ratings_analysis[
        test_ratings_analysis["is_unseen_movie"]
    ].copy()
    coldstart_test = coldstart_test[
        coldstart_test["movieId"].isin(multimodal_movie_index)
    ].copy()

    coldstart_test["user_idx"] = coldstart_test["userId"].map(user_to_index)
    coldstart_test["mm_movie_idx"] = coldstart_test["movieId"].map(
        multimodal_movie_index
    )
    coldstart_test["movie_idx"] = unk_movie_index

    print("Cold-start test interactions:", len(coldstart_test))
    print("Cold-start test movies:", coldstart_test["movieId"].nunique())
    print("Unique users:", coldstart_test["userId"].nunique())
    print("Missing user indices:", int(coldstart_test["user_idx"].isna().sum()))
    print("Missing content indices:", int(coldstart_test["mm_movie_idx"].isna().sum()))

    coldstart_users = coldstart_test["user_idx"].astype("int32").to_numpy()
    coldstart_movies = np.full(len(coldstart_test), unk_movie_index, dtype="int32")
    coldstart_content = coldstart_test["mm_movie_idx"].astype("int32").to_numpy()
    coldstart_targets = coldstart_test["rating"].astype("float32").to_numpy()

    coldstart_clip = aligned_clip[coldstart_content]
    coldstart_audio = aligned_audio[coldstart_content]

    coldstart_predictions = coldstart_model.predict(
        [coldstart_users, coldstart_movies, coldstart_clip, coldstart_audio],
        batch_size=cfg.batch_size,
        verbose=1,
    ).ravel()
    coldstart_predictions = np.clip(
        coldstart_predictions, cfg.rating_min, cfg.rating_max
    )

    cs_metrics = compute_regression_metrics(coldstart_targets, coldstart_predictions)
    global_subset_preds = np.full(
        len(coldstart_targets), train_global_mean, dtype="float32"
    )
    global_subset_metrics = compute_regression_metrics(
        coldstart_targets, global_subset_preds
    )

    mae_improvement = (
        (global_subset_metrics["mae"] - cs_metrics["mae"])
        / global_subset_metrics["mae"]
    ) * 100.0
    rmse_improvement = (
        (global_subset_metrics["rmse"] - cs_metrics["rmse"])
        / global_subset_metrics["rmse"]
    ) * 100.0

    print("\nCold-Start Test Results")
    print(f"MAE :  {cs_metrics['mae']:.4f}")
    print(f"RMSE:  {cs_metrics['rmse']:.4f}")
    print("\nGlobal Mean Baseline — Cold-Start Subset")
    print(f"MAE :  {global_subset_metrics['mae']:.4f}")
    print(f"RMSE:  {global_subset_metrics['rmse']:.4f}")
    print("\nCold-Start Improvement over Global Mean")
    print(f"MAE improvement :  {mae_improvement:.2f}%")
    print(f"RMSE improvement:  {rmse_improvement:.2f}%")

    # Step 8.5: Breakdown by test interaction count
    coldstart_movie_counts = (
        coldstart_test.groupby("movieId").size().rename("test_interactions").reset_index()
    )
    coldstart_test_analysis = coldstart_test.merge(
        coldstart_movie_counts, on="movieId", how="left"
    )
    coldstart_test_analysis["test_count_group"] = coldstart_test_analysis[
        "test_interactions"
    ].apply(assign_test_count_group)
    coldstart_test_analysis["prediction"] = coldstart_predictions

    group_rows: list[dict[str, object]] = []
    for group_name, group_df in coldstart_test_analysis.groupby("test_count_group"):
        group_metrics = compute_regression_metrics(
            group_df["rating"].to_numpy(), group_df["prediction"].to_numpy()
        )
        group_rows.append(
            {
                "Test Interaction Group": group_name,
                "Interactions": int(len(group_df)),
                "Movies": int(group_df["movieId"].nunique()),
                "MAE": group_metrics["mae"],
                "RMSE": group_metrics["rmse"],
            }
        )
    coldstart_group_results = pd.DataFrame(group_rows)
    print("\nCold-Start Performance by Test Interaction Count:")
    print(coldstart_group_results)

    coldstart_summary = {
        "Model": "Cold-Start Multimodal",
        "Test Interactions": int(len(coldstart_targets)),
        "Test Movies": int(coldstart_test["movieId"].nunique()),
        "MAE": cs_metrics["mae"],
        "RMSE": cs_metrics["rmse"],
        "Global Mean MAE": global_subset_metrics["mae"],
        "Global Mean RMSE": global_subset_metrics["rmse"],
        "MAE Improvement (%)": float(mae_improvement),
        "RMSE Improvement (%)": float(rmse_improvement),
    }
    return coldstart_summary, coldstart_group_results


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    evaluate_only: bool = False,
    save_metrics: bool = True,
) -> dict[str, object]:
    """Execute Stage 07 (Step 8): train/load cold-start model and evaluate unseen movies."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    train_ratings, val_ratings, test_ratings = load_checkpoint_splits(
        resolved_paths.checkpoint_dir
    )
    user_to_index, movie_to_index = build_or_load_id_mappings(
        train_ratings, resolved_paths, save_if_created=True
    )
    aligned_clip, aligned_audio, _, multimodal_movie_index = (
        build_or_load_aligned_embeddings(resolved_paths, save_if_created=True)
    )

    num_users = len(user_to_index)
    num_movies = len(movie_to_index)
    unk_movie_index = num_movies
    print("Known movie indices: 0 to", num_movies - 1)
    print("UNK movie index:", unk_movie_index)

    test_ratings_analysis, _ = analyze_seen_vs_unseen_test(
        train_ratings, test_ratings, multimodal_movie_index
    )

    checkpoint_path = (
        resolved_paths.model_checkpoint_dir / "coldstart_multimodal_best.keras"
    )
    if evaluate_only:
        existing_ckpt = resolve_existing_file(
            [
                checkpoint_path,
                resolved_paths.checkpoint_dir / "coldstart_multimodal_best.keras",
                resolved_paths.recommended_artifacts_dir
                / "coldstart_multimodal_model.keras",
            ],
            "Cold-start multimodal model checkpoint",
        )
        coldstart_model = keras.models.load_model(existing_ckpt)
    else:
        ensure_dir(resolved_paths.model_checkpoint_dir)
        coldstart_model = build_coldstart_multimodal_recommender(
            num_users=num_users,
            num_movies=num_movies,
            config=cfg,
        )
        coldstart_model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        coldstart_model.summary()

        train_users, train_movies, train_content, train_targets = (
            prepare_multimodal_arrays(
                train_ratings, user_to_index, movie_to_index, multimodal_movie_index
            )
        )
        train_movies_masked, train_mask = apply_coldstart_masking(
            train_movies,
            unk_movie_index=unk_movie_index,
            mask_rate=cfg.coldstart_mask_rate,
            seed=cfg.coldstart_train_seed,
        )
        print(
            f"Masked training interactions: {int(train_mask.sum()):,} "
            f"({train_mask.mean():.4f})"
        )

        val_users, val_movies, val_content, val_targets = prepare_multimodal_arrays(
            val_ratings, user_to_index, movie_to_index, multimodal_movie_index
        )
        val_movies_masked, val_mask = apply_coldstart_masking(
            val_movies,
            unk_movie_index=unk_movie_index,
            mask_rate=cfg.coldstart_mask_rate,
            seed=cfg.coldstart_val_seed,
        )
        print(
            f"Masked validation interactions: {int(val_mask.sum()):,} "
            f"({val_mask.mean():.4f})"
        )

        coldstart_train_dataset = create_multimodal_tf_dataset(
            train_users,
            train_movies_masked,
            train_content,
            train_targets,
            aligned_clip,
            aligned_audio,
            batch_size=cfg.batch_size,
            shuffle=True,
            repeat=True,
            clip_dim=cfg.clip_dim,
            audio_dim=cfg.audio_dim,
        )
        coldstart_val_dataset = create_multimodal_tf_dataset(
            val_users,
            val_movies_masked,
            val_content,
            val_targets,
            aligned_clip,
            aligned_audio,
            batch_size=cfg.batch_size,
            shuffle=False,
            repeat=True,
            clip_dim=cfg.clip_dim,
            audio_dim=cfg.audio_dim,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_mae",
                patience=2,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_mae",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
        ]

        coldstart_model.fit(
            coldstart_train_dataset,
            validation_data=coldstart_val_dataset,
            steps_per_epoch=int(np.ceil(len(train_targets) / cfg.batch_size)),
            validation_steps=int(np.ceil(len(val_targets) / cfg.batch_size)),
            epochs=cfg.coldstart_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    train_global_mean = float(train_ratings["rating"].mean())
    coldstart_summary, coldstart_group_results = evaluate_coldstart_subset(
        coldstart_model=coldstart_model,
        test_ratings_analysis=test_ratings_analysis,
        user_to_index=user_to_index,
        multimodal_movie_index=multimodal_movie_index,
        aligned_clip=aligned_clip,
        aligned_audio=aligned_audio,
        unk_movie_index=unk_movie_index,
        train_global_mean=train_global_mean,
        config=cfg,
    )

    if save_metrics:
        eval_label = (
            f"{coldstart_summary['Test Movies']:,} unseen movies / "
            f"{coldstart_summary['Test Interactions']:,} interactions"
        )
        save_model_metrics(
            metrics_dir=resolved_paths.metrics_dir,
            filename="coldstart_metrics.json",
            model_name="Cold-Start Multimodal",
            val_mae=None,
            val_rmse=None,
            test_mae=float(coldstart_summary["MAE"]),
            test_rmse=float(coldstart_summary["RMSE"]),
            evaluation=eval_label,
            extra={
                "Notes": f"Evaluation: {eval_label}",
                "Test Interactions": coldstart_summary["Test Interactions"],
                "Test Movies": coldstart_summary["Test Movies"],
                "Global Mean MAE": coldstart_summary["Global Mean MAE"],
                "Global Mean RMSE": coldstart_summary["Global Mean RMSE"],
                "MAE Improvement (%)": coldstart_summary["MAE Improvement (%)"],
                "RMSE Improvement (%)": coldstart_summary["RMSE Improvement (%)"],
            },
        )
        ensure_dir(resolved_paths.metrics_dir)
        coldstart_group_results.to_csv(
            resolved_paths.metrics_dir / "coldstart_group_results.csv",
            index=False,
        )
        print("Saved Cold-Start metrics to:", resolved_paths.metrics_dir)

    return {
        "coldstart_model": coldstart_model,
        "coldstart_summary": coldstart_summary,
        "coldstart_group_results": coldstart_group_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 07 (Step 8): Train and evaluate the Cold-Start Multimodal Model."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load existing cold-start model checkpoint and evaluate without retraining.",
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Skip writing metric JSON/CSV files to processed_v1/metrics/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(evaluate_only=args.eval_only, save_metrics=not args.no_save_metrics)
