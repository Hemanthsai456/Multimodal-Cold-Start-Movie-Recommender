"""Part 6 (Notebook Step 7): Multimodal Neural Recommender (User + Movie + CLIP + VGGish).

Aligns CLIP poster (512-D) and VGGish trailer-audio (128-D) embeddings across
19,232 catalog movies, constructs a memory-efficient ``tf.data.Dataset`` batch
pipeline, trains the multimodal recommender, evaluates validation/test performance,
and persists aligned mappings, model weights, and evaluation metrics.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam

from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    build_multimodal_network,
    build_or_load_aligned_embeddings,
    build_or_load_id_mappings,
    create_multimodal_tf_dataset,
    ensure_dir,
    load_checkpoint_splits,
    prepare_multimodal_arrays,
    print_regression_metrics,
    resolve_existing_file,
    save_model_metrics,
)


def build_multimodal_recommender(
    num_users: int,
    num_movies: int,
    config: PipelineConfig | None = None,
) -> keras.Model:
    """Build the 4-input Multimodal Neural Recommender for known training movies."""
    return build_multimodal_network(
        num_users=num_users,
        movie_vocab_size=num_movies,
        config=config,
    )


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    evaluate_only: bool = False,
    save_metrics: bool = True,
) -> dict[str, object]:
    """Execute Stage 06 (Step 7): align embeddings, train/load multimodal model, evaluate."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    train_ratings, val_ratings, test_ratings = load_checkpoint_splits(
        resolved_paths.checkpoint_dir
    )
    user_to_index, movie_to_index = build_or_load_id_mappings(
        train_ratings, resolved_paths, save_if_created=True
    )
    aligned_clip, aligned_audio, aligned_movie_ids, multimodal_movie_index = (
        build_or_load_aligned_embeddings(resolved_paths, save_if_created=True)
    )

    num_users = len(user_to_index)
    num_movies = len(movie_to_index)
    print(f"Aligned multimodal movies: {len(aligned_movie_ids):,}")
    print(f"Aligned CLIP shape:  {aligned_clip.shape}")
    print(f"Aligned Audio shape: {aligned_audio.shape}")

    train_users, train_movies, train_content, train_targets = prepare_multimodal_arrays(
        train_ratings, user_to_index, movie_to_index, multimodal_movie_index
    )
    print(
        f"Training coverage: {len(train_targets) / len(train_ratings) * 100:.2f}% "
        f"({len(train_targets):,} interactions)"
    )
    print("Index ranges:")
    print("  Users:  ", int(train_users.min()), "to", int(train_users.max()))
    print("  Movies: ", int(train_movies.min()), "to", int(train_movies.max()))
    print("  Content:", int(train_content.min()), "to", int(train_content.max()))
    print("  Ratings:", float(train_targets.min()), "to", float(train_targets.max()))

    val_users, val_movies, val_content, val_targets = prepare_multimodal_arrays(
        val_ratings, user_to_index, movie_to_index, multimodal_movie_index
    )
    test_users, test_movies, test_content, test_targets = prepare_multimodal_arrays(
        test_ratings, user_to_index, movie_to_index, multimodal_movie_index
    )

    train_dataset = create_multimodal_tf_dataset(
        train_users,
        train_movies,
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
    val_dataset = create_multimodal_tf_dataset(
        val_users,
        val_movies,
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
    test_dataset = create_multimodal_tf_dataset(
        test_users,
        test_movies,
        test_content,
        test_targets,
        aligned_clip,
        aligned_audio,
        batch_size=cfg.batch_size,
        shuffle=False,
        repeat=False,
        clip_dim=cfg.clip_dim,
        audio_dim=cfg.audio_dim,
    )

    val_steps = int(np.ceil(len(val_targets) / cfg.batch_size))
    test_steps = int(np.ceil(len(test_targets) / cfg.batch_size))
    checkpoint_path = resolved_paths.model_checkpoint_dir / "multimodal_best.keras"

    if evaluate_only:
        existing_ckpt = resolve_existing_file(
            [
                checkpoint_path,
                resolved_paths.checkpoint_dir / "multimodal_best.keras",
                resolved_paths.recommended_artifacts_dir / "multimodal_model.keras",
            ],
            "Multimodal model checkpoint",
        )
        multimodal_model = keras.models.load_model(existing_ckpt)
    else:
        ensure_dir(resolved_paths.model_checkpoint_dir)
        multimodal_model = build_multimodal_recommender(
            num_users=num_users,
            num_movies=num_movies,
            config=cfg,
        )
        multimodal_model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=[MeanAbsoluteError(name="mae")],
        )
        multimodal_model.summary()

        # Verify one training batch shape before fitting.
        sample_inputs, sample_targets = next(iter(train_dataset))
        print(
            "Batch check — Users:",
            sample_inputs[0].shape,
            "Movies:",
            sample_inputs[1].shape,
            "CLIP:",
            sample_inputs[2].shape,
            "Audio:",
            sample_inputs[3].shape,
            "Targets:",
            sample_targets.shape,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=2,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
        ]

        multimodal_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=cfg.multimodal_epochs,
            steps_per_epoch=int(np.ceil(len(train_targets) / cfg.batch_size)),
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )

    val_eval = multimodal_model.evaluate(val_dataset, steps=val_steps, verbose=1)
    test_eval = multimodal_model.evaluate(test_dataset, steps=test_steps, verbose=1)

    val_metrics = {
        "mae": float(val_eval[1]),
        "rmse": float(np.sqrt(val_eval[0])),
    }
    test_metrics = {
        "mae": float(test_eval[1]),
        "rmse": float(np.sqrt(test_eval[0])),
    }
    print_regression_metrics("Multimodal Model Performance", val_metrics, test_metrics)

    if save_metrics:
        save_model_metrics(
            metrics_dir=resolved_paths.metrics_dir,
            filename="multimodal_metrics.json",
            model_name="Multimodal",
            val_mae=val_metrics["mae"],
            val_rmse=val_metrics["rmse"],
            test_mae=test_metrics["mae"],
            test_rmse=test_metrics["rmse"],
            evaluation="Overall",
            extra={
                "Notes": "Multimodal Model (User + Movie + CLIP + VGGish)",
                "test_loss_mse": float(test_eval[0]),
            },
        )
        print("Saved Multimodal metrics to:", resolved_paths.metrics_dir)

    return {
        "multimodal_model": multimodal_model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "aligned_clip": aligned_clip,
        "aligned_audio": aligned_audio,
        "aligned_movie_ids": aligned_movie_ids,
        "multimodal_movie_index": multimodal_movie_index,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 06 (Step 7): Train and evaluate the Multimodal Neural Recommender."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load existing multimodal checkpoint and evaluate without retraining.",
    )
    parser.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="Skip writing metric JSON files to processed_v1/metrics/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(evaluate_only=args.eval_only, save_metrics=not args.no_save_metrics)
