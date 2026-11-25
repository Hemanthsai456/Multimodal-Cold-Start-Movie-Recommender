"""Part 5 (Notebook Step 6.3): Neural Collaborative Filtering (NCF) baseline.

Builds training-only user/movie index mappings, trains the collaborative NCF
neural baseline on seen-movie interactions, evaluates validation/test MAE and RMSE,
and saves mappings, best model checkpoint, and metrics under ``processed_v1/``.
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
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Input
from tensorflow.keras.optimizers import Adam

from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    build_or_load_id_mappings,
    compute_regression_metrics,
    ensure_dir,
    load_checkpoint_splits,
    print_regression_metrics,
    resolve_existing_file,
    save_model_metrics,
)


def prepare_ncf_data(
    df: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> pd.DataFrame:
    """Attach integer embedding indices for users and movies."""
    data = df.copy()
    data["user_idx"] = data["userId"].map(user_to_index)
    data["movie_idx"] = data["movieId"].map(movie_to_index)
    return data


def extract_ncf_arrays(
    df: pd.DataFrame,
    drop_unseen_movies: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert mapped NCF dataframe into ``(user_idx, movie_idx, rating)`` numpy arrays."""
    subset = df.dropna(subset=["user_idx", "movie_idx"]).copy() if drop_unseen_movies else df
    users = subset["user_idx"].astype("int32").to_numpy()
    movies = subset["movie_idx"].astype("int32").to_numpy()
    targets = subset["rating"].astype("float32").to_numpy()
    return users, movies, targets


def build_ncf_model(
    num_users: int,
    num_movies: int,
    embedding_dim: int = 32,
    dropout_rate: float = 0.2,
) -> keras.Model:
    """Create the Neural Collaborative Filtering baseline model."""
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_dim)(user_input)
    movie_embedding = Embedding(num_movies, embedding_dim)(movie_input)

    x = Concatenate()([user_embedding, movie_embedding])
    x = Dense(64, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)
    return keras.Model([user_input, movie_input], output)


def evaluate_ncf_model(
    ncf_model: keras.Model,
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: PipelineConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate trained NCF model on seen-movie validation and test splits."""
    cfg = config or PipelineConfig()
    val_users, val_movies, val_targets = val_arrays
    test_users, test_movies, test_targets = test_arrays

    val_pred = ncf_model.predict(
        [val_users, val_movies], batch_size=cfg.eval_batch_size, verbose=1
    ).ravel()
    val_pred = np.clip(val_pred, cfg.rating_min, cfg.rating_max)
    val_metrics = compute_regression_metrics(val_targets, val_pred)

    test_pred = ncf_model.predict(
        [test_users, test_movies], batch_size=cfg.eval_batch_size, verbose=1
    ).ravel()
    test_pred = np.clip(test_pred, cfg.rating_min, cfg.rating_max)
    test_metrics = compute_regression_metrics(test_targets, test_pred)

    print_regression_metrics("NCF Baseline Performance", val_metrics, test_metrics)
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    evaluate_only: bool = False,
    save_metrics: bool = True,
) -> dict[str, object]:
    """Execute Stage 05 (Baseline 3 - NCF): map IDs, train/load model, and evaluate."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    train_ratings, val_ratings, test_ratings = load_checkpoint_splits(
        resolved_paths.checkpoint_dir
    )
    user_to_index, movie_to_index = build_or_load_id_mappings(
        train_ratings, resolved_paths, save_if_created=True
    )
    num_users = len(user_to_index)
    num_movies = len(movie_to_index)
    print(f"Users in training:  {num_users:,}")
    print(f"Movies in training: {num_movies:,}")

    train_ncf = prepare_ncf_data(train_ratings, user_to_index, movie_to_index)
    val_ncf = prepare_ncf_data(val_ratings, user_to_index, movie_to_index)
    test_ncf = prepare_ncf_data(test_ratings, user_to_index, movie_to_index)

    print("Validation unseen movies:", int(val_ncf["movie_idx"].isna().sum()))
    print("Test unseen movies:", int(test_ncf["movie_idx"].isna().sum()))

    train_users, train_movies, train_targets = extract_ncf_arrays(
        train_ncf, drop_unseen_movies=False
    )
    val_arrays = extract_ncf_arrays(val_ncf, drop_unseen_movies=True)
    test_arrays = extract_ncf_arrays(test_ncf, drop_unseen_movies=True)

    checkpoint_path = resolved_paths.model_checkpoint_dir / "ncf_best.keras"
    if evaluate_only:
        existing_ckpt = resolve_existing_file(
            [checkpoint_path, resolved_paths.checkpoint_dir / "ncf_best.keras"],
            "NCF checkpoint",
        )
        ncf_model = keras.models.load_model(existing_ckpt)
    else:
        ensure_dir(resolved_paths.model_checkpoint_dir)
        ncf_model = build_ncf_model(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=cfg.embedding_dim,
            dropout_rate=cfg.dropout_rate,
        )
        ncf_model.compile(
            optimizer=Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        ncf_model.summary()

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
                patience=2,
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

        ncf_model.fit(
            x=[train_users, train_movies],
            y=train_targets,
            validation_data=([val_arrays[0], val_arrays[1]], val_arrays[2]),
            epochs=cfg.ncf_epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

    eval_res = evaluate_ncf_model(ncf_model, val_arrays, test_arrays, config=cfg)

    if save_metrics:
        save_model_metrics(
            metrics_dir=resolved_paths.metrics_dir,
            filename="ncf_metrics.json",
            model_name="NCF",
            val_mae=eval_res["val_metrics"]["mae"],
            val_rmse=eval_res["val_metrics"]["rmse"],
            test_mae=eval_res["test_metrics"]["mae"],
            test_rmse=eval_res["test_metrics"]["rmse"],
            evaluation="Overall",
            extra={"Notes": "Neural collaborative filtering"},
        )
        print("Saved NCF metrics to:", resolved_paths.metrics_dir)

    return {
        "ncf_model": ncf_model,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        **eval_res,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 05 (Step 6.3): Train and evaluate Neural Collaborative Filtering baseline."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load existing ncf_best.keras checkpoint and evaluate without retraining.",
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
