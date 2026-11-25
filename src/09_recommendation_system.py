"""Part 9 (Notebook Steps 10, 11 & 12): Artifact packaging and recommendation system.

Step 10 packages trained models, ID mappings, aligned multimodal embeddings,
movie metadata, user seen-movie history, and configuration into
``recommended_artifacts/`` (optionally applying Keras compatibility patching via
``fix_keras_models.remove_quantization_config``).

Steps 11 & 12 delegate inference to ``MovieRecommender`` (``recommender.py``) to
perform movie search, single-movie rating prediction (routing automatically
between the Multimodal Model and Cold-Start Multimodal Model), personalized Top-N
recommendations, genre filtering, hybrid movie-based recommendations, edge-case
checks, and demonstration result exports under ``recommendation_results/``.

See ``FUTURE_IMPROVEMENTS`` at the end of this module for planned V2+ enhancements.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from fix_keras_models import patch_model
from recommender import MovieRecommender
from src.config import PipelineConfig, ProjectPaths
from src.utils import (
    build_or_load_aligned_embeddings,
    build_or_load_id_mappings,
    ensure_dir,
    load_json,
    resolve_existing_file,
    save_json,
    save_pickle,
)

# ---------------------------------------------------------------------------
# Future Improvements Roadmap (see also README.md)
# ---------------------------------------------------------------------------
FUTURE_IMPROVEMENTS: list[str] = [
    "1. Online Multimodal Feature Extraction: Extract CLIP poster and VGGish audio "
    "embeddings dynamically for newly released or user-uploaded movies outside the V1 catalog.",
    "2. True User Cold-Start Handling: Support unknown/new users via onboarding taste "
    "quizzes, demographic/context priors, or session-based preference representations.",
    "3. Graceful Missing-Modality Fallback: Handle partial modalities (e.g., poster-only "
    "without trailer audio) using learned modality dropout or cross-modal imputation instead "
    "of requiring both CLIP and VGGish embeddings.",
    "4. Ranking & Beyond-Accuracy Evaluation: Add top-K ranking metrics (NDCG@K, Recall@K, "
    "HitRate@K, MAP@K) alongside rating regression metrics (MAE/RMSE) plus diversity/novelty checks.",
    "5. Richer Multimodal Fusion & Metadata: Incorporate text synopsis/review embeddings, "
    "genre/cast/director encoders, and cross-attention or gated fusion layers.",
    "6. Production API & CI Testing: Serve predictions via FastAPI/ONNX/TF-Serving with "
    "automated unit tests, artifact schema validation, and experiment tracking.",
]


def export_recommendation_artifacts(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    apply_keras_compat_patch: bool = True,
) -> Path:
    """Step 10: Package trained models, mappings, embeddings, and history for serving."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()
    artifact_dir = ensure_dir(resolved_paths.recommended_artifacts_dir)

    train_path = resolved_paths.checkpoint_dir / "train_ratings.parquet"
    movies_path = resolved_paths.checkpoint_dir / "multimodal_movies.parquet"
    for req in (train_path, movies_path):
        if not req.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {req}. Run Parts 1 & 3 first."
            )

    train_ratings = pd.read_parquet(train_path)
    multimodal_movies = pd.read_parquet(movies_path)[["movieId", "title", "genres"]]

    user_to_index, movie_to_index = build_or_load_id_mappings(
        train_ratings, resolved_paths, save_if_created=True
    )
    aligned_clip, aligned_audio, aligned_movie_ids, multimodal_movie_index = (
        build_or_load_aligned_embeddings(resolved_paths, save_if_created=True)
    )

    mm_src = resolve_existing_file(
        [
            resolved_paths.model_checkpoint_dir / "multimodal_best.keras",
            resolved_paths.checkpoint_dir / "multimodal_best.keras",
        ],
        "trained multimodal model checkpoint",
    )
    cs_src = resolve_existing_file(
        [
            resolved_paths.model_checkpoint_dir / "coldstart_multimodal_best.keras",
            resolved_paths.checkpoint_dir / "coldstart_multimodal_best.keras",
        ],
        "trained cold-start multimodal model checkpoint",
    )

    mm_dest = artifact_dir / "multimodal_model.keras"
    cs_dest = artifact_dir / "coldstart_multimodal_model.keras"
    shutil.copy2(mm_src, mm_dest)
    shutil.copy2(cs_src, cs_dest)

    if apply_keras_compat_patch:
        patch_model(mm_dest)
        patch_model(cs_dest)

    save_pickle(user_to_index, artifact_dir / "user_to_index.pkl")
    save_pickle(movie_to_index, artifact_dir / "movie_to_index.pkl")
    save_pickle(multimodal_movie_index, artifact_dir / "multimodal_movie_index.pkl")

    user_seen_movies = train_ratings.groupby("userId")["movieId"].apply(set).to_dict()
    save_pickle(user_seen_movies, artifact_dir / "user_seen_movies.pkl")

    np.save(artifact_dir / "aligned_clip.npy", aligned_clip)
    np.save(artifact_dir / "aligned_audio.npy", aligned_audio)
    np.save(
        artifact_dir / "aligned_movie_ids.npy",
        np.asarray(aligned_movie_ids, dtype=np.int32),
    )
    multimodal_movies.to_parquet(
        artifact_dir / "multimodal_movies.parquet", index=False
    )

    num_users = len(user_to_index)
    num_movies = len(movie_to_index)
    artifact_config = {
        "num_users": int(num_users),
        "num_movies": int(num_movies),
        "clip_dim": int(aligned_clip.shape[1]),
        "audio_dim": int(aligned_audio.shape[1]),
        "embedding_dim": int(cfg.embedding_dim),
        "unk_movie_index": int(num_movies),
        "num_multimodal_movies": int(len(multimodal_movie_index)),
    }
    save_json(artifact_config, artifact_dir / "config.json")

    # Reload verification
    reloaded = MovieRecommender(artifact_dir)
    loaded_cfg = load_json(artifact_dir / "config.json")
    print("\n===== STEP 10 ARTIFACT EXPORT & RELOAD VERIFICATION =====")
    print("Artifact directory:", artifact_dir)
    print("Stats:", reloaded.stats)
    print("Config:", json.dumps(loaded_cfg, indent=4))
    return artifact_dir


def run_step11_demo_and_edge_cases(
    rec: MovieRecommender,
    results_dir: Path,
    config: PipelineConfig | None = None,
    save_results: bool = True,
) -> dict[str, object]:
    """Step 11: Run standard recommendation demo, edge cases, and save outputs."""
    cfg = config or PipelineConfig()
    demo_user_id = cfg.demo_user_id
    known_movie_id = cfg.demo_known_movie_id
    coldstart_movie_id = cfg.demo_coldstart_movie_id

    seen_count = len(rec.user_seen_movies.get(demo_user_id, set()))
    known_title = rec.movie_meta_map.get(known_movie_id, {}).get("title", "Unknown")
    cold_title = rec.movie_meta_map.get(coldstart_movie_id, {}).get("title", "Unknown")

    print("\n=== DEMO USER & MOVIE SELECTION ===")
    print(f"User ID: {demo_user_id} (Training Ratings Count: {seen_count})")
    print(f"Known Movie ID: {known_movie_id} - Title: {known_title}")
    print(f"Cold-Start Movie ID: {coldstart_movie_id} - Title: {cold_title}")

    print("\n=== A. Personalized Top-10 Recommendations ===")
    top10_recs = rec.recommend_movies(
        user_id=demo_user_id,
        top_n=cfg.default_top_n,
        batch_size=cfg.inference_batch_size,
    )
    print(top10_recs.to_string(index=False))

    print("\n=== B. Single Rating Prediction for Known Movie ===")
    pred_known = rec.predict_movie_rating(user_id=demo_user_id, movie_id=known_movie_id)
    print(json.dumps(pred_known, indent=4))

    print(
        "\n=== C. Single Rating Prediction for True Cold-Start Movie "
        "(0 Training Ratings) ==="
    )
    pred_cold = rec.predict_movie_rating(
        user_id=demo_user_id, movie_id=coldstart_movie_id
    )
    print(json.dumps(pred_cold, indent=4))

    print("\n=== Edge Case 1: Invalid User ID ===")
    edge_user = rec.predict_movie_rating(user_id=99999999, movie_id=known_movie_id)
    print(edge_user)

    print("\n=== Edge Case 2: Movie Without Multimodal Embeddings ===")
    edge_movie = rec.predict_movie_rating(user_id=demo_user_id, movie_id=99999999)
    print(edge_movie)

    edge_cases = {
        "invalid_user": edge_user,
        "missing_embeddings_movie": edge_movie,
    }

    if save_results:
        ensure_dir(results_dir)
        export_cols = [
            "Rank",
            "Movie",
            "Genres",
            "Predicted Rating",
            "Model Type",
            "Training Status",
        ]
        top10_to_save = top10_recs[
            [c for c in export_cols if c in top10_recs.columns]
        ]
        top10_to_save.to_csv(results_dir / "top10_recommendations_user1.csv", index=False)
        save_json(pred_known, results_dir / "single_prediction_known_movie.json")
        save_json(pred_cold, results_dir / "single_prediction_coldstart_movie.json")
        save_json(edge_cases, results_dir / "edge_cases_test.json")

        summary_lines = [
            "Step 11 Recommendation System Summary:",
            "--------------------------------------------",
            f"Demo User ID: {demo_user_id}",
            f"Known Movie ID: {known_movie_id} ({known_title})",
            (
                f"Cold-Start Movie ID: {coldstart_movie_id} ({cold_title}) - "
                "Training Rating Count: 0"
            ),
            "",
            "Top-10 Recommendations:",
            top10_to_save.to_string(index=False),
            "",
            f"Known Movie Model: {pred_known.get('model_type')}",
            f"Cold-Start Movie Model: {pred_cold.get('model_type')}",
            (
                "Validation Tests Passed: "
                f"{edge_user.get('status') == 'error' and edge_movie.get('status') == 'error'}"
            ),
        ]
        (results_dir / "demo_summary.txt").write_text(
            "\n".join(summary_lines), encoding="utf-8"
        )
        print("\nSaved Step 11 demonstration outputs to:", results_dir)

    return {
        "top10_recs": top10_recs,
        "pred_known": pred_known,
        "pred_cold": pred_cold,
        "edge_cases": edge_cases,
    }


def run_step12_extended_demo(
    rec: MovieRecommender,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """Step 12: Run search, second-user known/cold-start inference, and hybrid demo."""
    cfg = config or PipelineConfig()
    print("\n=== Step 12: Movie Search ('Fight') ===")
    search_res = rec.search_movies("Fight", top_n=cfg.default_top_n)
    print(search_res[["movieId", "title", "genres"]].to_string(index=False))

    user_id = cfg.extended_user_id
    known_movie_id = cfg.extended_known_movie_id
    cold_movie_id = cfg.extended_coldstart_movie_id

    print(f"\n=== Known Movie Prediction (User {user_id}, Movie {known_movie_id}) ===")
    ext_known = rec.predict_movie_rating(user_id, known_movie_id)
    print(json.dumps(ext_known, indent=4))

    print(
        f"\n=== Cold-Start Movie Prediction (User {user_id}, Movie {cold_movie_id}) ==="
    )
    ext_cold = rec.predict_movie_rating(user_id, cold_movie_id)
    print(json.dumps(ext_cold, indent=4))

    print(f"\n=== Top-10 Recommendations for User {user_id} ===")
    ext_top10 = rec.recommend_movies(
        user_id, top_n=cfg.default_top_n, batch_size=cfg.inference_batch_size
    )
    print(ext_top10.to_string(index=False))

    print(
        f"\n=== Hybrid Movie-Based Recommendations (User {user_id}, Seed {known_movie_id}) ==="
    )
    hybrid_recs = rec.recommend_from_movie(
        user_id=user_id,
        movie_id=known_movie_id,
        top_n=5,
        batch_size=cfg.inference_batch_size,
    )
    print(hybrid_recs.to_string(index=False))

    return {
        "search_res": search_res,
        "ext_known": ext_known,
        "ext_cold": ext_cold,
        "ext_top10": ext_top10,
        "hybrid_recs": hybrid_recs,
    }


def print_future_improvements() -> None:
    """Display planned V2+ future improvements for the recommendation system."""
    print("\n===== FUTURE IMPROVEMENTS =====")
    for item in FUTURE_IMPROVEMENTS:
        print(f"- {item}")


def run(
    paths: ProjectPaths | None = None,
    config: PipelineConfig | None = None,
    export_artifacts: bool = False,
    patch_keras: bool = True,
    save_results: bool = True,
    run_extended: bool = True,
    query: str | None = None,
    user_id: int | None = None,
    movie_id: int | None = None,
    seed_movie_id: int | None = None,
    genre: str | None = None,
    top_n: int | None = None,
) -> dict[str, object]:
    """Execute Stage 09 (Steps 10–12): optionally export artifacts and run inference."""
    resolved_paths = paths or ProjectPaths()
    cfg = config or PipelineConfig()

    if export_artifacts:
        export_recommendation_artifacts(
            paths=resolved_paths,
            config=cfg,
            apply_keras_compat_patch=patch_keras,
        )

    rec = MovieRecommender(resolved_paths.recommended_artifacts_dir)
    print("Loaded MovieRecommender stats:", rec.stats)

    k = top_n or cfg.default_top_n
    if any(v is not None for v in (query, user_id, movie_id, seed_movie_id, genre)):
        output: dict[str, object] = {}
        if query is not None:
            matches = rec.search_movies(query, top_n=k)
            print(f"\nSearch results for '{query}':")
            print(matches.to_string(index=False))
            output["search"] = matches
        if user_id is not None and movie_id is not None:
            rating_pred = rec.predict_movie_rating(user_id, movie_id)
            print("\nSingle movie rating prediction:")
            print(json.dumps(rating_pred, indent=4))
            output["prediction"] = rating_pred
        elif user_id is not None and seed_movie_id is not None:
            hybrid = rec.recommend_from_movie(
                user_id=user_id,
                movie_id=seed_movie_id,
                top_n=k,
                batch_size=cfg.inference_batch_size,
            )
            print(f"\nHybrid recommendations for User {user_id} from Movie {seed_movie_id}:")
            print(hybrid.to_string(index=False))
            output["hybrid_recommendations"] = hybrid
        elif user_id is not None:
            recs = rec.recommend_movies(
                user_id=user_id,
                top_n=k,
                batch_size=cfg.inference_batch_size,
                genre=genre,
            )
            print(f"\nTop-{k} recommendations for User {user_id} (genre={genre}):")
            print(recs.to_string(index=False))
            output["recommendations"] = recs
        print_future_improvements()
        return output

    step11_res = run_step11_demo_and_edge_cases(
        rec=rec,
        results_dir=resolved_paths.recommendation_results_dir,
        config=cfg,
        save_results=save_results,
    )
    step12_res = (
        run_step12_extended_demo(rec=rec, config=cfg) if run_extended else {}
    )
    print_future_improvements()
    return {
        "step11": step11_res,
        "step12": step12_res,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 09 (Steps 10–12): Export artifacts and run recommendation inference."
    )
    parser.add_argument(
        "--export-artifacts",
        action="store_true",
        help="Run Step 10 to package trained models and artifacts into recommended_artifacts/.",
    )
    parser.add_argument(
        "--no-patch-keras",
        action="store_true",
        help="Skip Keras Embedding quantization_config compatibility patching during export.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Skip writing Step 11 demo outputs to recommendation_results/.",
    )
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="Skip Step 12 extended demonstration.",
    )
    parser.add_argument("--query", type=str, default=None, help="Search movie titles/genres.")
    parser.add_argument("--user-id", type=int, default=None, help="Target MovieLens User ID.")
    parser.add_argument("--movie-id", type=int, default=None, help="Target Movie ID to score.")
    parser.add_argument(
        "--seed-movie-id",
        type=int,
        default=None,
        help="Seed Movie ID for hybrid movie-based recommendations.",
    )
    parser.add_argument("--genre", type=str, default=None, help="Optional genre filter.")
    parser.add_argument("--top-n", type=int, default=None, help="Number of recommendations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        export_artifacts=args.export_artifacts,
        patch_keras=not args.no_patch_keras,
        save_results=not args.no_save_results,
        run_extended=not args.skip_extended,
        query=args.query,
        user_id=args.user_id,
        movie_id=args.movie_id,
        seed_movie_id=args.seed_movie_id,
        genre=args.genre,
        top_n=args.top_n,
    )
