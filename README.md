# Multimodal Cold-Start Movie Recommender

A multimodal movie recommendation system that combines collaborative filtering with visual and audio representations to address the cold-start problem for movies with little or no historical rating data.

## Overview

Traditional recommendation systems depend heavily on user–item interaction history. This creates a problem when a movie is new or has very few ratings.

This project explores whether pretrained multimodal representations can provide useful movie information even when collaborative feedback is unavailable.

The system combines:

- MovieLens 20M user–movie ratings
- CLIP poster embeddings (512-D)
- VGGish trailer-audio embeddings (128-D)
- Neural Collaborative Filtering (NCF)
- Multimodal neural recommendation
- A dedicated cold-start recommendation model with an explicit unknown-movie (`UNK`) embedding
- A modular Python pipeline (`src/`), standalone inference wrapper (`recommender.py`), and Streamlit web interface (`app.py`)

---

## Problem

A newly released or rarely rated movie has limited interaction history.

A conventional collaborative recommender may therefore struggle to estimate user preference for that movie.

This project addresses this by incorporating movie content representations:

**Poster → CLIP embedding (512-D)**

**Trailer audio → VGGish embedding (128-D)**

These representations are combined with user and movie representations to produce predicted ratings.

---

## Approach

The project evaluates several approaches:

1. Global Mean Baseline
2. User + Movie Bias
3. Neural Collaborative Filtering (NCF)
4. Multimodal Recommender
5. Cold-Start Multimodal Recommender

The proposed multimodal model combines:

- User embedding (`32-D`)
- Movie embedding (`32-D`)
- Projected CLIP poster embedding (`512 → 32`)
- Projected VGGish trailer-audio embedding (`128 → 32`)

For movies that were completely unseen during training, a dedicated cold-start model is trained with 20% random movie-ID masking to an explicit `UNK` movie representation together with the available visual and audio embeddings.

---

## Dataset

### MovieLens 20M

Used for:

- User ratings (`20,000,263` total source ratings across `138,493` users and `27,278` movies)
- Movie metadata (`movieId`, `title`, `genres`)
- Per-user chronological train (80%) / validation (10%) / test (10%) splitting

Dataset:

https://grouplens.org/datasets/movielens/

### M³L Multimodal Features

Used for:

- CLIP poster embeddings (`25,037` movies, `512-D`)
- VGGish trailer-audio embeddings (`19,340` movies, `128-D`)

Source:

https://zenodo.org/records/18499145

The project uses the provided pretrained embeddings rather than training CLIP or VGGish from scratch.

---

## Dataset Alignment

After aligning MovieLens with the multimodal features:

- Multimodal catalog movies (with both CLIP and VGGish): **19,232** (`19,014` with ratings + `218` catalog movies with zero ratings)
- Aligned multimodal ratings: **18,788,341** across **138,493** users (`99.2865%` sparsity)
- Chronological split: **14,975,947** train / **1,871,633** validation / **1,940,761** test interactions
- CLIP embedding dimension: **512**
- VGGish embedding dimension: **128**
- Movies present in training interactions: **16,517**
- Test movies completely unseen during training: **2,107**
- Cold-start test interactions: **3,630**

The rating data is highly sparse, motivating the use of content information alongside collaborative signals.

---

## Results

### Overall Evaluation

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Global Mean | 0.8228 | 1.0399 | 0.8267 | 1.0445 |
| User + Movie Bias | 0.6665 | 0.8741 | 0.6799 | 0.8926 |
| NCF | 0.8211 | 1.0383 | 0.8255 | 1.0433 |
| Multimodal | **0.6258** | **0.8136** | **0.6373** | **0.8319** |

The multimodal model improved over the User + Movie Bias baseline by approximately:

- **6.27% Test MAE improvement** (`0.6799 → 0.6373`)
- **6.80% Test RMSE improvement** (`0.8926 → 0.8319`)

### Cold-Start Evaluation

For movies completely unseen during training:

- Movies: **2,107**
- Test interactions: **3,630**
- Cold-start MAE: **0.6731**
- Cold-start RMSE: **0.8948**

Compared with the global-mean baseline evaluated on the same cold-start subset (MAE `0.8401`, RMSE `1.1243`):

- MAE improvement: **19.88%**
- RMSE improvement: **20.41%**

Cold-start performance also improves monotonically with test interaction density (`1–2`: `0.7023` MAE, `3–5`: `0.6385` MAE, `6+`: `0.5840` MAE). The cold-start and overall multimodal results are evaluated on different populations and therefore should not be directly compared as equivalent metrics.

---

## Modular Pipeline Architecture

The monolithic research notebook ([`notebooks/01_Multimodal_ColdStart_Movie.ipynb`](notebooks/01_Multimodal_ColdStart_Movie.ipynb)) is preserved untouched as the primary reference, and its full workflow is modularized into nine standalone runnable Python stages under `src/` backed by shared configuration (`src/config.py`) and utilities (`src/utils.py`):

| Part | Notebook Scope | Module (`src/`) | Primary Inputs | Outputs / Saved Artifacts |
|---:|---|---|---|---|
| **1** | Step 1 & Step 2: Libraries, Raw Data, Profiling & Alignment | `01_data_loading_and_preprocessing.py` | `../../../ml-20m/`, `../../../IMG_clip-image/`, `../../../AUD_vggish/` | `../../../processed_v1/`: `multimodal_movies.parquet`, `multimodal_ratings.parquet`, `clip_*.npy`, `vggish_*.npy`, `source_dataset_stats.json` |
| **2** | Step 3: Recommendation Data Analysis & EDA | `02_data_analysis_and_eda.py` | `../../../processed_v1/` (`multimodal_ratings.parquet` or split parquets, `multimodal_movies.parquet`) | Summary stats, Top-10 movies, EDA plots in `../../../processed_v1/plots/` |
| **3** | Step 4 (& Step 5): Chronological Split, Split EDA & Checkpoints | `03_train_val_test_split.py` | `../../../processed_v1/` (`multimodal_ratings.parquet`, `multimodal_movies.parquet`) | `train/val/test_ratings.parquet`, `test_movie_sparsity_analysis.parquet`, `sparse_feedback_summary.csv`, `dataset_metadata.json`, split plots |
| **4** | Step 6.1 & 6.2: Baselines 1 & 2 (Global Mean, User + Movie Bias) | `04_baseline_mean_and_bias.py` | `../../../processed_v1/` split parquets | `../../../processed_v1/metrics/`: `global_mean_metrics.json`, `bias_model_metrics.json` |
| **5** | Step 6.3: Baseline 3 (Neural Collaborative Filtering) | `05_baseline_ncf.py` | `../../../processed_v1/` split parquets | `../../../processed_v1/mappings/` (`user_to_index.pkl`, `movie_to_index.pkl`), `models/ncf_best.keras`, `metrics/ncf_metrics.json` |
| **6** | Step 7: Multimodal Neural Recommender | `06_multimodal_model.py` | Splits, raw embeddings, ID mappings | `mappings/aligned_*.npy`, `mappings/multimodal_movie_index.pkl`, `models/multimodal_best.keras`, `metrics/multimodal_metrics.json` |
| **7** | Step 8: Cold-Start Multimodal Model & Seen/Unseen Eval | `07_coldstart_model.py` | Splits, ID mappings, aligned embeddings | `models/coldstart_multimodal_best.keras`, `metrics/coldstart_metrics.json`, `metrics/coldstart_group_results.csv` |
| **8** | Step 9: Model Comparison & Evaluation | `08_model_comparison_and_evaluation.py` | `../../../processed_v1/metrics/*.json` (or `--eval-saved-models`) | `final_results.csv` (`metrics/` & `recommendation_results/`), 5 comparison plots in `../../../processed_v1/plots/` |
| **9** | Steps 10, 11 & 12: Artifact Packaging & Recommendation System | `09_recommendation_system.py` | `recommended_artifacts/` (and `../../../processed_v1/` when `--export-artifacts`) | `recommended_artifacts/` (Step 10), `recommendation_results/` demo outputs & edge-case reports (Steps 11–12) |

### Environment Setup

Activate the project virtual environment (`.venv`) and install the core dependencies before running any `src/` scripts or the Streamlit app:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

> **Additional runtime packages:** `requirements.txt` pins the core ML/data stack (`tensorflow`, `keras`, `numpy`, `pandas`, `pyarrow`, `scikit-learn`). Install `matplotlib` if running the plotting stages (`02_data_analysis_and_eda.py`, `08_model_comparison_and_evaluation.py`), and install `streamlit` if launching `app.py` (`pip install matplotlib streamlit`).

### Running the Modular Scripts

Each stage can be executed independently from the repository root once its upstream data/checkpoints exist:

- **Ready immediately (existing `recommended_artifacts/`):** Run `09_recommendation_system.py` for CLI inference/demos or `streamlit run app.py` for the UI without retraining or external raw data.
- **Ready from existing `processed_v1/` checkpoints (no raw CSV/JSON reload):** Run `02_data_analysis_and_eda.py` (falls back to existing split parquets), `03_train_val_test_split.py`, `04_baseline_mean_and_bias.py`, `--eval-only` on `05`–`07`, or `08_model_comparison_and_evaluation.py --eval-saved-models`.
- **Full end-to-end training from scratch:** Requires raw `ml-20m/`, `IMG_clip-image/`, and `AUD_vggish/` datasets outside the repository (`01` → `09`).

```bash
# Data loading, EDA, and chronological splitting
python src/01_data_loading_and_preprocessing.py
python src/02_data_analysis_and_eda.py
python src/03_train_val_test_split.py

# Baseline and multimodal model training (Stages 04–07)
python src/04_baseline_mean_and_bias.py
python src/05_baseline_ncf.py                 # or --eval-only
python src/06_multimodal_model.py             # or --eval-only
python src/07_coldstart_model.py              # or --eval-only

# Model comparison & evaluation
python src/08_model_comparison_and_evaluation.py
# Or re-evaluate existing saved checkpoints directly without retraining:
python src/08_model_comparison_and_evaluation.py --eval-saved-models

# Recommendation inference demo (uses existing recommended_artifacts/)
python src/09_recommendation_system.py
# Optional flags for Stage 09:
#   --export-artifacts       Run Step 10 to rebuild & patch recommended_artifacts/
#   --user-id 25 --top-n 10  Generate personalized Top-N recommendations
#   --user-id 25 --movie-id 2959          Predict rating for a single movie
#   --user-id 25 --seed-movie-id 2959     Hybrid movie-based recommendations
#   --query "Fight"                       Search catalog by ID, title, or genre
```

### Launching the Streamlit App

```bash
streamlit run app.py
```

If Keras models trained in another environment encounter an `Embedding` `quantization_config` deserialization issue under local Keras `3.15.1`, run:

```bash
python fix_keras_models.py
```

---

## Recommendation System

The inference system (`recommender.py` & `09_recommendation_system.py`) supports:

### Personalized Top-N Recommendation

Given a known MovieLens user, the system:

1. Builds a candidate pool of multimodal movies (with optional genre filtering).
2. Removes movies already seen by the user during training.
3. Routes known movies through the multimodal model.
4. Routes training-unseen movies through the cold-start model (`UNK` movie index).
5. Ranks movies by predicted rating (clipped to `[0.5, 5.0]`).
6. Returns the Top-N recommendations.

### Hybrid Movie-Based Recommendation

Given a user and a selected movie (`MovieRecommender.recommend_from_movie`), the system combines:

- **70%** personalized rating prediction from the trained multimodal / cold-start models
- **30%** average cosine similarity across CLIP poster and VGGish trailer-audio embeddings

### Individual Movie Prediction

The system can also estimate a user's predicted rating for a selected movie. The model is automatically selected based on whether the movie was present during training:

```text
User
  ↓
Movie Candidate
  ↓
Known during training?
  ├── Yes → Multimodal Model
  └── No  → Cold-Start Multimodal Model
                    ↓
             Predicted Rating
```

---

## Project Structure

```text
Leveraging-Poster-and-Trailer-Audio-Embeddings-for-Cold-Start-Movie-Recommendation/
│
├── notebooks/
│   └── 01_Multimodal_ColdStart_Movie.ipynb     # Original end-to-end reference notebook (preserved)
│
├── src/                                        # Modularized 9-part Python pipeline
│   ├── __init__.py
│   ├── config.py                               # Centralized paths & hyperparameters
│   ├── utils.py                                # Shared I/O, metrics, mappings, dataset & model builders
│   ├── 01_data_loading_and_preprocessing.py
│   ├── 02_data_analysis_and_eda.py
│   ├── 03_train_val_test_split.py
│   ├── 04_baseline_mean_and_bias.py
│   ├── 05_baseline_ncf.py
│   ├── 06_multimodal_model.py
│   ├── 07_coldstart_model.py
│   ├── 08_model_comparison_and_evaluation.py
│   └── 09_recommendation_system.py
│
├── recommended_artifacts/                      # Saved models, mappings, aligned embeddings & config
│   ├── multimodal_model.keras
│   ├── coldstart_multimodal_model.keras
│   ├── user_to_index.pkl
│   ├── movie_to_index.pkl
│   ├── multimodal_movie_index.pkl
│   ├── user_seen_movies.pkl
│   ├── aligned_clip.npy
│   ├── aligned_audio.npy
│   ├── aligned_movie_ids.npy
│   ├── multimodal_movies.parquet
│   └── config.json
│
├── recommendation_results/                     # Saved Step 11 demo & edge-case outputs
│   ├── demo_summary.txt
│   ├── edge_cases_test.json
│   ├── single_prediction_coldstart_movie.json
│   ├── single_prediction_known_movie.json
│   └── top10_recommendations_user1.csv
│
├── app.py                                      # Streamlit UI entrypoint
├── ui/                                         # Modular Streamlit UI pages & styling
│   ├── __init__.py
│   ├── styles.py                               # Custom CSS styling
│   ├── recommendations.py                      # Recommendations page
│   ├── cold_start.py                           # Cold-start prediction page
│   └── about.py                                # About Model page
├── recommender.py                              # Reusable MovieRecommender inference class
├── fix_keras_models.py                         # Keras Embedding quantization_config compatibility patcher
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

External raw datasets (`ml-20m/`, `IMG_clip-image/`, `AUD_vggish/`) and large intermediate checkpoints (`processed_v1/`) remain outside the repository root (`../../../` relative to `notebooks/`). The local `.venv/` environment is intentionally excluded from version control.

---

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* PyArrow
* Matplotlib
* Streamlit
* MovieLens 20M
* CLIP embeddings
* VGGish embeddings
* Jupyter Notebook

---

## Limitations

This V1 implementation has several limitations:

* Recommendations require a known MovieLens user (`user_to_index`).
* The multimodal inference catalog is limited to movies with both CLIP and VGGish embeddings (`19,232` movies).
* Missing modalities are not automatically replaced or imputed.
* Arbitrary external movies without pre-extracted embeddings are outside the scope of V1.
* CLIP and VGGish embeddings are used as fixed pretrained features rather than fine-tuned end-to-end.
* Evaluation focuses on rating prediction error (`MAE`, `RMSE`) rather than top-K ranking or diversity metrics.

---

## Data Usage & Attribution

This project uses publicly available datasets/features for academic and research purposes.

Please refer to the original sources for their respective terms of use and attribution requirements.

### MovieLens

MovieLens dataset provided by GroupLens Research.

Citation:

> Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 19.

### M³L

Multimodal movie features are obtained from the M³L dataset/resource.

The original dataset source and associated terms should be followed when using or redistributing the data.

No raw copyrighted movie posters, trailers, or other movie media are redistributed with this repository.

---

## Project Status

**V1 — Modularized & Completed**

Current version includes:

* Dataset integration and multimodal feature alignment
* Chronological per-user splitting and cold-start sparsity analysis
* Baseline, Multimodal, and Cold-Start Multimodal model experimentation
* Model comparison and evaluation reporting
* Saved inference engine (`recommender.py`) and Streamlit interface (`app.py`)
* Modular 9-part runnable Python pipeline (`src/`) preserving the original notebook (`notebooks/01_Multimodal_ColdStart_Movie.ipynb`)

---

## Future Improvements

Planned enhancements for V2+ (also documented in [`src/09_recommendation_system.py`](src/09_recommendation_system.py)):

1. **Online Multimodal Feature Extraction**: Extract CLIP poster and VGGish trailer-audio embeddings dynamically for newly uploaded or newly released movies outside the fixed V1 catalog.
2. **True User Cold-Start Handling**: Support new/unknown users via taste onboarding quizzes, session-based history, or fallback content-popularity priors.
3. **Graceful Missing-Modality Fallback**: Handle partial modalities (e.g., poster available without trailer audio) via modality dropout during training or cross-modal embedding projection rather than dropping movies.
4. **Top-K Ranking & Beyond-Accuracy Metrics**: Add ranking evaluation (`NDCG@K`, `Recall@K`, `HitRate@K`, `MAP@K`) alongside `MAE`/`RMSE`, plus catalog coverage, novelty, and calibration checks.
5. **Richer Content & Fusion Architectures**: Incorporate plot synopsis/review text embeddings, genre/cast/director encodings, and gated or cross-attention fusion layers instead of plain feature concatenation.
6. **Production Serving & Automated Testing**: Package inference behind a FastAPI/TF-Serving service with automated unit/integration tests, deterministic data/model versioning, and experiment tracking.
