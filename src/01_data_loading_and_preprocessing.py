"""Part 1 (Notebook Steps 1 & 2): Data loading, profiling, and multimodal alignment.

Loads MovieLens 20M ratings and movie metadata alongside pretrained CLIP poster
(512-D) and VGGish trailer-audio (128-D) embeddings, checks ID integrity, aligns
the datasets, and saves intermediate handoff files under ``processed_v1/``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.config import ProjectPaths
from src.utils import ensure_dir, save_json


def load_movielens_data(
    ratings_path: Path,
    movies_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens ratings and movies CSV files."""
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies CSV not found: {movies_path}")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    return ratings, movies


def profile_source_data(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict[str, int]:
    """Inspect MovieLens data schemas, unique counts, and ID consistency."""
    print("\nRatings head:\n", ratings.head())
    print("\nMovies head:\n", movies.head())

    print("\nRatings Info:")
    ratings.info()
    print("\nMovies Info:")
    movies.info()

    print("\nUnique values in Ratings:\n", ratings.nunique())
    print("\nUnique values in Movies:\n", movies.nunique())

    rating_movie_ids = set(ratings["movieId"])
    movie_ids = set(movies["movieId"])
    missing_movies = rating_movie_ids - movie_ids

    print("Movie IDs in ratings:", len(rating_movie_ids))
    print("Movie IDs in movies:", len(movie_ids))
    print("Missing movie IDs:", len(missing_movies))

    return {
        "total_ratings_source": int(len(ratings)),
        "source_users": int(ratings["userId"].nunique()),
        "source_movies": int(movies["movieId"].nunique()),
        "missing_movie_ids": int(len(missing_movies)),
    }


def load_embedding_jsons(folder_path: Path, label: str) -> dict[int, list[float]]:
    """Load movie-ID -> embedding-vector dictionaries from all JSON files in ``folder_path``."""
    if not folder_path.exists():
        raise FileNotFoundError(f"{label} folder not found: {folder_path}")

    features: dict[int, list[float]] = {}
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = folder_path / filename
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for movie_id, embedding in data.items():
                features[int(movie_id)] = embedding

    print(f"{label} movies:", len(features))
    if features:
        first_movie_id = next(iter(features))
        print(f"Sample {label} Movie ID:", first_movie_id)
        print(f"{label} Embedding length:", len(features[first_movie_id]))
    return features


def align_multimodal_datasets(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    clip_features: dict[int, list[float]],
    audio_features: dict[int, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Align MovieLens metadata and ratings with CLIP and VGGish modalities.

    ``multimodal_movies`` retains all movies present in MovieLens metadata with both
    CLIP and audio embeddings (19,232 movies, including 218 cold-start catalog movies
    with 0 ratings), while ``multimodal_ratings`` retains interactions for rated movies
    across all four sources (19,014 movies / 18,788,341 ratings).
    """
    movie_ids = set(movies["movieId"])
    clip_ids = set(clip_features.keys())
    audio_ids = set(audio_features.keys())
    rating_movie_ids = set(ratings["movieId"])

    multimodal_catalog_ids = movie_ids & clip_ids & audio_ids
    multimodal_rated_ids = multimodal_catalog_ids & rating_movie_ids

    print("Movies:", len(movie_ids))
    print("Ratings movies:", len(rating_movie_ids))
    print("CLIP:", len(clip_ids))
    print("Audio:", len(audio_ids))
    print("Movies with metadata + CLIP + Audio:", len(multimodal_catalog_ids))
    print("All four (with ratings):", len(multimodal_rated_ids))

    multimodal_movies = movies[movies["movieId"].isin(multimodal_catalog_ids)][
        ["movieId", "title", "genres"]
    ].copy()
    multimodal_ratings = ratings[ratings["movieId"].isin(multimodal_rated_ids)].copy()

    print("\nFinal DATASET")
    print("Multimodal catalog movies:", multimodal_movies["movieId"].nunique())
    print("Original ratings:", len(ratings))
    print("Multimodal ratings:", len(multimodal_ratings))
    print("Users:", multimodal_ratings["userId"].nunique())
    print("Rated multimodal movies:", multimodal_ratings["movieId"].nunique())

    alignment_stats = {
        "aligned_movies": int(len(multimodal_movies)),
        "movies_with_ratings": int(multimodal_ratings["movieId"].nunique()),
        "movies_without_ratings": int(
            len(multimodal_movies) - multimodal_ratings["movieId"].nunique()
        ),
        "clip_movies": int(len(clip_features)),
        "audio_movies": int(len(audio_features)),
    }
    return multimodal_movies, multimodal_ratings, alignment_stats


def save_preprocessing_outputs(
    multimodal_movies: pd.DataFrame,
    multimodal_ratings: pd.DataFrame,
    clip_features: dict[int, list[float]],
    audio_features: dict[int, list[float]],
    source_stats: dict[str, int],
    checkpoint_dir: Path,
) -> None:
    """Save aligned datasets, embedding arrays, and source stats to ``checkpoint_dir``."""
    ensure_dir(checkpoint_dir)

    movies_path = checkpoint_dir / "multimodal_movies.parquet"
    ratings_path = checkpoint_dir / "multimodal_ratings.parquet"
    multimodal_movies.to_parquet(movies_path, index=False)
    multimodal_ratings.to_parquet(ratings_path, index=False)

    clip_movie_ids = np.array(list(clip_features.keys()), dtype=np.int32)
    clip_embeddings = np.asarray(list(clip_features.values()), dtype=np.float32)
    np.save(checkpoint_dir / "clip_movie_ids.npy", clip_movie_ids)
    np.save(checkpoint_dir / "clip_embeddings.npy", clip_embeddings)

    vggish_movie_ids = np.array(list(audio_features.keys()), dtype=np.int32)
    vggish_embeddings = np.asarray(list(audio_features.values()), dtype=np.float32)
    np.save(checkpoint_dir / "vggish_movie_ids.npy", vggish_movie_ids)
    np.save(checkpoint_dir / "vggish_embeddings.npy", vggish_embeddings)

    stats_payload = {
        **source_stats,
        "clip_dimension": int(clip_embeddings.shape[1]),
        "audio_dimension": int(vggish_embeddings.shape[1]),
    }
    save_json(stats_payload, checkpoint_dir / "source_dataset_stats.json")

    print("Saved Step 1 & 2 preprocessing outputs to:", checkpoint_dir)


def run(
    paths: ProjectPaths | None = None,
    save_outputs: bool = True,
) -> dict[str, object]:
    """Execute Stage 01 (Steps 1 & 2): data loading, profiling, and alignment."""
    resolved_paths = paths or ProjectPaths()
    ratings, movies = load_movielens_data(
        resolved_paths.ratings_csv, resolved_paths.movies_csv
    )
    source_stats = profile_source_data(ratings, movies)
    clip_features = load_embedding_jsons(resolved_paths.clip_folder, "CLIP")
    audio_features = load_embedding_jsons(resolved_paths.audio_folder, "Audio")

    multimodal_movies, multimodal_ratings, alignment_stats = align_multimodal_datasets(
        ratings, movies, clip_features, audio_features
    )
    combined_stats = {**source_stats, **alignment_stats}

    if save_outputs:
        save_preprocessing_outputs(
            multimodal_movies=multimodal_movies,
            multimodal_ratings=multimodal_ratings,
            clip_features=clip_features,
            audio_features=audio_features,
            source_stats=combined_stats,
            checkpoint_dir=resolved_paths.checkpoint_dir,
        )

    return {
        "multimodal_movies": multimodal_movies,
        "multimodal_ratings": multimodal_ratings,
        "clip_features": clip_features,
        "audio_features": audio_features,
        "source_stats": combined_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 01 (Steps 1 & 2): Load raw datasets and align multimodal features."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run profiling and alignment in memory without writing checkpoint files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(save_outputs=not args.no_save)
