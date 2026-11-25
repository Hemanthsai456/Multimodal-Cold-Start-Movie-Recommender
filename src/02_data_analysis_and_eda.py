"""Part 2 (Notebook Step 3): Recommendation data analysis and exploratory data analysis.

Computes user/movie interaction counts, matrix sparsity, rating distributions,
top-rated movie statistics, and generates exploratory distribution plots.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ProjectPaths
from src.utils import load_multimodal_inputs, save_or_show_figure


def compute_interaction_statistics(multimodal_ratings: pd.DataFrame) -> dict[str, object]:
    """Compute user/movie rating distribution summaries and matrix sparsity."""
    print("Multimodal ratings shape:", multimodal_ratings.shape)
    num_users = int(multimodal_ratings["userId"].nunique())
    num_movies = int(multimodal_ratings["movieId"].nunique())
    num_ratings = int(len(multimodal_ratings))

    print("Unique users:", num_users)
    print("Unique movies:", num_movies)
    print("Total ratings:", num_ratings)

    ratings_per_movie = multimodal_ratings.groupby("movieId").size()
    ratings_per_user = multimodal_ratings.groupby("userId").size()

    print("\nRatings per movie summary:")
    print(ratings_per_movie.describe())
    print("\nRatings per user summary:")
    print(ratings_per_user.describe())

    possible_interactions = num_users * num_movies
    sparsity = 1.0 - (num_ratings / possible_interactions)

    print(f"\nPossible user-movie interactions: {possible_interactions:,}")
    print(f"Observed ratings: {num_ratings:,}")
    print(f"Sparsity: {sparsity:.4%}")
    print("\nRating summary:\n", multimodal_ratings["rating"].describe())

    return {
        "num_users": num_users,
        "num_movies": num_movies,
        "num_ratings": num_ratings,
        "possible_interactions": possible_interactions,
        "sparsity": sparsity,
        "ratings_per_movie": ratings_per_movie,
        "ratings_per_user": ratings_per_user,
    }


def get_top_rated_movies(
    multimodal_ratings: pd.DataFrame,
    multimodal_movies: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the ``top_n`` most frequently rated movies joined with titles."""
    top_movies = (
        multimodal_ratings.groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_movies_df = top_movies.reset_index(name="rating_count").merge(
        multimodal_movies[["movieId", "title"]],
        on="movieId",
        how="left",
    )
    print(f"\nTop {top_n} most rated movies:")
    print(top_movies_df)
    return top_movies_df


def generate_eda_plots(
    multimodal_ratings: pd.DataFrame,
    ratings_per_movie: pd.Series,
    ratings_per_user: pd.Series,
    plots_dir: Path | None = None,
    show_plots: bool = False,
) -> None:
    """Create Step 3 exploratory plots for rating values, movie counts, and user counts."""
    rating_counts = multimodal_ratings["rating"].value_counts().sort_index()

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    rating_counts.plot(kind="bar", ax=ax1)
    ax1.set_title("Distribution of Movie Ratings")
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Number of Ratings")
    ax1.tick_params(axis="x", rotation=0)
    save_or_show_figure(
        fig1,
        plots_dir / "step03_rating_distribution.png" if plots_dir else None,
        show_plots=show_plots,
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(ratings_per_movie, bins=30)
    ax2.set_title("Distribution of Ratings per Movie")
    ax2.set_xlabel("Number of Ratings")
    ax2.set_ylabel("Number of Movies")
    ax2.set_xscale("log")
    save_or_show_figure(
        fig2,
        plots_dir / "step03_ratings_per_movie.png" if plots_dir else None,
        show_plots=show_plots,
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.hist(ratings_per_user, bins=32)
    ax3.set_title("Distribution of Ratings per User")
    ax3.set_xlabel("Number of Ratings")
    ax3.set_ylabel("Number of Users")
    ax3.set_xscale("log")
    save_or_show_figure(
        fig3,
        plots_dir / "step03_ratings_per_user.png" if plots_dir else None,
        show_plots=show_plots,
    )


def run(
    paths: ProjectPaths | None = None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> dict[str, object]:
    """Execute Stage 02 (Step 3): dataset profiling and exploratory visualizations."""
    resolved_paths = paths or ProjectPaths()
    multimodal_ratings, multimodal_movies = load_multimodal_inputs(
        resolved_paths.checkpoint_dir
    )
    stats = compute_interaction_statistics(multimodal_ratings)
    top_movies_df = get_top_rated_movies(multimodal_ratings, multimodal_movies, top_n=10)

    plots_dir = resolved_paths.plots_dir if save_plots else None
    generate_eda_plots(
        multimodal_ratings=multimodal_ratings,
        ratings_per_movie=stats["ratings_per_movie"],
        ratings_per_user=stats["ratings_per_user"],
        plots_dir=plots_dir,
        show_plots=show_plots,
    )

    return {
        "stats": stats,
        "top_movies_df": top_movies_df,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 02 (Step 3): Recommendation data analysis and EDA."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display matplotlib windows interactively in addition to saving PNGs.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Skip saving EDA figures to processed_v1/plots/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(show_plots=args.show_plots, save_plots=not args.no_save_plots)
