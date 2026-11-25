"""Shared I/O, metric, mapping, dataset, and model utilities for the pipeline."""

import importlib
from pathlib import Path
import json
import pickle
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PipelineConfig, ProjectPaths


def import_step_module(module_name: str):
    """Dynamically import a numbered step module from ``src`` (e.g. ``'04_baseline_mean_and_bias'``)."""
    return importlib.import_module(f"src.{module_name}")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (including parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    """Serialize ``data`` as pretty-printed UTF-8 JSON."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path) -> Any:
    """Load UTF-8 JSON from ``path``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    """Serialize ``obj`` to ``path`` using pickle."""
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load a pickled Python object from ``path``."""
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve_existing_file(candidates: list[Path], description: str) -> Path:
    """Return the first existing path from ``candidates`` or raise ``FileNotFoundError``."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked: {searched}")


def load_multimodal_inputs(checkpoint_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned multimodal ratings and movies from ``checkpoint_dir``.

    Uses ``multimodal_ratings.parquet`` produced by ``01_data_loading_and_preprocessing.py``
    when available, and falls back to concatenating existing split checkpoints if
    only Step 5 split outputs exist.
    """
    movies_path = checkpoint_dir / "multimodal_movies.parquet"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"Missing {movies_path}. Run 01_data_loading_and_preprocessing.py first."
        )
    multimodal_movies = pd.read_parquet(movies_path)

    ratings_path = checkpoint_dir / "multimodal_ratings.parquet"
    if ratings_path.exists():
        multimodal_ratings = pd.read_parquet(ratings_path)
    else:
        split_paths = [
            checkpoint_dir / "train_ratings.parquet",
            checkpoint_dir / "val_ratings.parquet",
            checkpoint_dir / "test_ratings.parquet",
        ]
        if all(p.exists() for p in split_paths):
            multimodal_ratings = pd.concat(
                [pd.read_parquet(p) for p in split_paths],
                ignore_index=True,
            )
        else:
            raise FileNotFoundError(
                f"Missing {ratings_path} (and train/val/test split parquets). "
                "Run 01_data_loading_and_preprocessing.py first."
            )

    return multimodal_ratings, multimodal_movies


def load_checkpoint_splits(
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test rating splits from ``checkpoint_dir``."""
    train_path = checkpoint_dir / "train_ratings.parquet"
    val_path = checkpoint_dir / "val_ratings.parquet"
    test_path = checkpoint_dir / "test_ratings.parquet"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing split checkpoint {p}. Run 03_train_val_test_split.py first."
            )
    return (
        pd.read_parquet(train_path),
        pd.read_parquet(val_path),
        pd.read_parquet(test_path),
    )


def load_raw_embedding_arrays(
    checkpoint_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw CLIP and VGGish embedding arrays and their movie ID vectors."""
    required = {
        "clip_movie_ids": checkpoint_dir / "clip_movie_ids.npy",
        "clip_embeddings": checkpoint_dir / "clip_embeddings.npy",
        "vggish_movie_ids": checkpoint_dir / "vggish_movie_ids.npy",
        "vggish_embeddings": checkpoint_dir / "vggish_embeddings.npy",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {name} file at {path}. Run 01_data_loading_and_preprocessing.py first."
            )
    return (
        np.load(required["clip_movie_ids"]),
        np.load(required["clip_embeddings"]),
        np.load(required["vggish_movie_ids"]),
        np.load(required["vggish_embeddings"]),
    )


def build_or_load_id_mappings(
    train_ratings: pd.DataFrame,
    paths: ProjectPaths,
    save_if_created: bool = True,
) -> tuple[dict[int, int], dict[int, int]]:
    """Load training ID mappings or create them deterministically from ``train_ratings``."""
    user_candidates = [
        paths.mappings_dir / "user_to_index.pkl",
        paths.recommended_artifacts_dir / "user_to_index.pkl",
    ]
    movie_candidates = [
        paths.mappings_dir / "movie_to_index.pkl",
        paths.recommended_artifacts_dir / "movie_to_index.pkl",
    ]
    if user_candidates[0].exists() and movie_candidates[0].exists():
        return load_pickle(user_candidates[0]), load_pickle(movie_candidates[0])

    # Build exclusively from training interactions to avoid data leakage.
    user_ids = train_ratings["userId"].unique()
    movie_ids = train_ratings["movieId"].unique()
    user_to_index = {int(uid): idx for idx, uid in enumerate(user_ids)}
    movie_to_index = {int(mid): idx for idx, mid in enumerate(movie_ids)}

    if save_if_created:
        ensure_dir(paths.mappings_dir)
        save_pickle(user_to_index, paths.mappings_dir / "user_to_index.pkl")
        save_pickle(movie_to_index, paths.mappings_dir / "movie_to_index.pkl")

    return user_to_index, movie_to_index


def build_or_load_aligned_embeddings(
    paths: ProjectPaths,
    save_if_created: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Return ``(aligned_clip, aligned_audio, aligned_movie_ids, multimodal_movie_index)``.

    Prefers pre-aligned handoff files in ``paths.mappings_dir`` (or
    ``paths.recommended_artifacts_dir``), otherwise aligns raw CLIP and VGGish
    arrays from ``paths.checkpoint_dir`` by intersecting movie IDs.
    """
    m_dir = paths.mappings_dir
    files = (
        m_dir / "aligned_clip.npy",
        m_dir / "aligned_audio.npy",
        m_dir / "aligned_movie_ids.npy",
        m_dir / "multimodal_movie_index.pkl",
    )
    if all(p.exists() for p in files):
        return (
            np.load(files[0]),
            np.load(files[1]),
            np.load(files[2]),
            load_pickle(files[3]),
        )

    clip_movie_ids, clip_embeddings, audio_movie_ids, audio_embeddings = (
        load_raw_embedding_arrays(paths.checkpoint_dir)
    )
    clip_index = {int(mid): idx for idx, mid in enumerate(clip_movie_ids)}
    audio_index = {int(mid): idx for idx, mid in enumerate(audio_movie_ids)}

    aligned_movie_ids_list = sorted(set(clip_index.keys()) & set(audio_index.keys()))
    aligned_clip = np.stack(
        [clip_embeddings[clip_index[mid]] for mid in aligned_movie_ids_list]
    ).astype("float32")
    aligned_audio = np.stack(
        [audio_embeddings[audio_index[mid]] for mid in aligned_movie_ids_list]
    ).astype("float32")
    aligned_movie_ids = np.array(aligned_movie_ids_list, dtype=np.int32)
    multimodal_movie_index = {
        int(mid): idx for idx, mid in enumerate(aligned_movie_ids_list)
    }

    if save_if_created:
        ensure_dir(m_dir)
        np.save(files[0], aligned_clip)
        np.save(files[1], aligned_audio)
        np.save(files[2], aligned_movie_ids)
        save_pickle(multimodal_movie_index, files[3])

    return aligned_clip, aligned_audio, aligned_movie_ids, multimodal_movie_index


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    include_r2: bool = False,
) -> dict[str, float]:
    """Compute MAE, RMSE, and optionally R-squared for rating predictions."""
    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if include_r2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    return metrics


def print_regression_metrics(
    title: str,
    val_metrics: dict[str, float] | None,
    test_metrics: dict[str, float],
) -> None:
    """Pretty-print validation and test regression metrics."""
    print(title)
    print("-" * 35)
    if val_metrics is not None:
        print(f"Validation MAE:  {val_metrics['mae']:.4f}")
        print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
        if "r2" in val_metrics:
            print(f"Validation R2:   {val_metrics['r2']:.4f}")
    print(f"Test MAE:        {test_metrics['mae']:.4f}")
    print(f"Test RMSE:       {test_metrics['rmse']:.4f}")
    if "r2" in test_metrics:
        print(f"Test R2:         {test_metrics['r2']:.4f}")


def save_model_metrics(
    metrics_dir: Path,
    filename: str,
    model_name: str,
    val_mae: float | None,
    val_rmse: float | None,
    test_mae: float,
    test_rmse: float,
    evaluation: str = "Overall",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist standardized model evaluation results for Part 8 comparison."""
    ensure_dir(metrics_dir)
    record: dict[str, Any] = {
        "Model": model_name,
        "Validation MAE": val_mae,
        "Validation RMSE": val_rmse,
        "Test MAE": float(test_mae),
        "Test RMSE": float(test_rmse),
        "Evaluation": evaluation,
    }
    if extra:
        record.update(extra)
    out_path = metrics_dir / filename
    save_json(record, out_path)
    return out_path


def prepare_multimodal_arrays(
    df: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    multimodal_movie_index: dict[int, int],
    filter_known_movies: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map dataframe rows to ``(users, movies, content_indices, targets)`` numpy arrays."""
    subset = df[df["movieId"].isin(multimodal_movie_index)].copy()
    if filter_known_movies:
        subset = subset[
            subset["userId"].isin(user_to_index) & subset["movieId"].isin(movie_to_index)
        ].copy()

    users = subset["userId"].map(user_to_index).astype("int32").to_numpy()
    movies = subset["movieId"].map(movie_to_index).astype("int32").to_numpy()
    content_indices = (
        subset["movieId"].map(multimodal_movie_index).astype("int32").to_numpy()
    )
    targets = subset["rating"].astype("float32").to_numpy()
    return users, movies, content_indices, targets


def multimodal_batch_generator(
    users: np.ndarray,
    movies: np.ndarray,
    content_indices: np.ndarray,
    targets: np.ndarray,
    aligned_clip: np.ndarray,
    aligned_audio: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True,
) -> Iterator[tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """Yield mini-batches with on-the-fly CLIP and VGGish content lookup."""
    n = len(targets)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_content = content_indices[batch_idx]
        yield (
            (
                users[batch_idx],
                movies[batch_idx],
                aligned_clip[batch_content],
                aligned_audio[batch_content],
            ),
            targets[batch_idx],
        )


def create_multimodal_tf_dataset(
    users: np.ndarray,
    movies: np.ndarray,
    content_indices: np.ndarray,
    targets: np.ndarray,
    aligned_clip: np.ndarray,
    aligned_audio: np.ndarray,
    batch_size: int = 1024,
    shuffle: bool = True,
    repeat: bool = True,
    clip_dim: int = 512,
    audio_dim: int = 128,
):
    """Build a prefetching ``tf.data.Dataset`` wrapping ``multimodal_batch_generator``."""
    import tensorflow as tf

    output_signature = (
        (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None, clip_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, audio_dim), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: multimodal_batch_generator(
            users=users,
            movies=movies,
            content_indices=content_indices,
            targets=targets,
            aligned_clip=aligned_clip,
            aligned_audio=aligned_audio,
            batch_size=batch_size,
            shuffle=shuffle,
        ),
        output_signature=output_signature,
    )
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_multimodal_network(
    num_users: int,
    movie_vocab_size: int,
    config: PipelineConfig | None = None,
):
    """Construct the 4-input multimodal rating regression model in Keras.

    Used by both Step 7 (``movie_vocab_size = num_movies``) and Step 8
    cold-start training (``movie_vocab_size = num_movies + 1`` with explicit UNK slot).
    """
    from tensorflow import keras
    from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Flatten

    cfg = config or PipelineConfig()

    user_input = keras.Input(shape=(1,), dtype="int32")
    movie_input = keras.Input(shape=(1,), dtype="int32")
    clip_input = keras.Input(shape=(cfg.clip_dim,), dtype="float32")
    audio_input = keras.Input(shape=(cfg.audio_dim,), dtype="float32")

    user_emb = Flatten()(
        Embedding(input_dim=num_users, output_dim=cfg.embedding_dim)(user_input)
    )
    movie_emb = Flatten()(
        Embedding(input_dim=movie_vocab_size, output_dim=cfg.embedding_dim)(movie_input)
    )

    clip_emb = Dense(cfg.embedding_dim, activation="relu")(clip_input)
    audio_emb = Dense(cfg.embedding_dim, activation="relu")(audio_input)

    x = Concatenate()([user_emb, movie_emb, clip_emb, audio_emb])
    x = Dense(64, activation="relu")(x)
    x = Dropout(cfg.dropout_rate)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1)(x)

    return keras.Model(
        inputs=[user_input, movie_input, clip_input, audio_input],
        outputs=output,
    )


def save_or_show_figure(
    fig,
    save_path: Path | None = None,
    show_plots: bool = False,
) -> None:
    """Optionally save a matplotlib figure to disk and show or close it."""
    import matplotlib.pyplot as plt

    if save_path is not None:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
