"""Central configuration for paths and hyperparameters across all pipeline stages.

Relative path mapping matches the original ``notebooks/01_Multimodal_ColdStart_Movie.ipynb``
layout:
    - Raw datasets and intermediate checkpoints live three levels above ``notebooks/``
      (i.e. ``REPO_ROOT.parent.parent``: ``ml-20m/``, ``IMG_clip-image/``,
      ``AUD_vggish/``, and ``processed_v1/``).
    - Saved deployment artifacts and demonstration results live at the repository root
      (``recommended_artifacts/`` and ``recommendation_results/``).
"""

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EXTERNAL_DATA_ROOT: Path = REPO_ROOT.parent.parent


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths used by the 9 modular pipeline stages."""

    repo_root: Path = REPO_ROOT
    external_root: Path = EXTERNAL_DATA_ROOT

    # Raw input datasets (read-only sources outside the repository)
    ml20m_dir: Path = field(default_factory=lambda: EXTERNAL_DATA_ROOT / "ml-20m")
    clip_folder: Path = field(default_factory=lambda: EXTERNAL_DATA_ROOT / "IMG_clip-image")
    audio_folder: Path = field(default_factory=lambda: EXTERNAL_DATA_ROOT / "AUD_vggish")

    # Intermediate checkpoints & handoff files (../../../processed_v1 from notebooks/)
    checkpoint_dir: Path = field(default_factory=lambda: EXTERNAL_DATA_ROOT / "processed_v1")

    # Deployment artifacts and demo outputs inside the repository
    recommended_artifacts_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "recommended_artifacts"
    )
    recommendation_results_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "recommendation_results"
    )

    @property
    def ratings_csv(self) -> Path:
        return self.ml20m_dir / "ratings.csv"

    @property
    def movies_csv(self) -> Path:
        return self.ml20m_dir / "movies.csv"

    @property
    def model_checkpoint_dir(self) -> Path:
        return self.checkpoint_dir / "models"

    @property
    def mappings_dir(self) -> Path:
        return self.checkpoint_dir / "mappings"

    @property
    def metrics_dir(self) -> Path:
        return self.checkpoint_dir / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.checkpoint_dir / "plots"


@dataclass(frozen=True)
class PipelineConfig:
    """Hyperparameters and constants matching the V1 experimental setup."""

    # Chronological per-user split ratios (Step 4)
    train_ratio: float = 0.80
    val_ratio: float = 0.10

    # Embedding & architecture dimensions (Steps 6–8)
    clip_dim: int = 512
    audio_dim: int = 128
    embedding_dim: int = 32
    dropout_rate: float = 0.20

    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 1024
    eval_batch_size: int = 4096
    ncf_epochs: int = 10
    multimodal_epochs: int = 15
    coldstart_epochs: int = 15

    # Cold-start UNK masking (Step 8)
    coldstart_mask_rate: float = 0.20
    coldstart_train_seed: int = 42
    coldstart_val_seed: int = 123

    # Valid MovieLens rating bounds
    rating_min: float = 0.5
    rating_max: float = 5.0

    # Inference defaults (Steps 11–12)
    inference_batch_size: int = 2048
    default_top_n: int = 10
    demo_user_id: int = 1
    demo_known_movie_id: int = 1
    demo_coldstart_movie_id: int = 3561
    extended_user_id: int = 25
    extended_known_movie_id: int = 2959
    extended_coldstart_movie_id: int = 6501
