"""
Configuration management using Pydantic.
All configs are typed and validated.
"""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
import torch


class DataConfig(BaseModel):
    """Configuration for synthetic data generation."""
    model_config = {"extra": "forbid"}

    seed: int = 42
    n_users: int = 5000
    n_venues: int = 300
    n_events: int = 30000
    n_interest_dims: int = 16
    n_categories: int = 20
    n_geo_cells: int = 100  # 10x10 grid
    horizon_days: int = 14

    # Friend graph params (stochastic block model)
    n_communities: int = 10
    intra_community_prob: float = 0.15
    inter_community_prob: float = 0.01

    # Calendar params
    avg_busy_blocks_per_day: float = 3.0
    busy_block_duration_min: int = 60
    busy_block_duration_max: int = 240

    # Action generation params
    base_view_prob: float = 0.3
    base_like_prob: float = 0.1
    base_save_prob: float = 0.05
    base_intent_prob: float = 0.03
    base_attend_prob: float = 0.02
    availability_boost: float = 3.0  # Strong boost when available
    social_proof_boost: float = 2.0  # Boost per friend intent
    distance_penalty_scale: float = 0.1

    # Output paths
    data_dir: Path = Path("data")


class SmallDataConfig(DataConfig):
    """Smaller config for quick testing."""
    n_users: int = 500
    n_venues: int = 50
    n_events: int = 3000
    n_communities: int = 5
    # Higher action rates for small mode to get enough training data
    base_view_prob: float = 0.5
    base_like_prob: float = 0.2
    base_save_prob: float = 0.1
    base_intent_prob: float = 0.08
    base_attend_prob: float = 0.05


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    model_config = {"extra": "forbid"}

    # Two-tower model
    user_embed_dim: int = 64
    event_embed_dim: int = 64
    tower_hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])

    # Reranker
    reranker_hidden_dims: list[int] = Field(default_factory=lambda: [64, 32])
    reranker_dropout: float = 0.1

    # GNN
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1


class TrainConfig(BaseModel):
    """Configuration for training."""
    model_config = {"extra": "forbid"}

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Two-tower training
    tt_batch_size: int = 1024
    tt_epochs: int = 20
    tt_lr: float = 1e-3
    tt_weight_decay: float = 1e-5
    tt_neg_samples: int = 10

    # Reranker training
    rr_batch_size: int = 512
    rr_epochs: int = 15
    rr_lr: float = 1e-3

    # GNN training
    gnn_batch_size: int = 512
    gnn_epochs: int = 30
    gnn_lr: float = 1e-3
    gnn_neg_samples: int = 5

    # Train/val/test split (by day)
    train_days: tuple[int, int] = (1, 10)
    val_days: tuple[int, int] = (11, 12)
    test_days: tuple[int, int] = (13, 14)

    # Output paths
    models_dir: Path = Path("models")
    runs_dir: Path = Path("runs")


class SmallTrainConfig(TrainConfig):
    """Smaller config for quick testing."""
    tt_epochs: int = 5
    rr_epochs: int = 5
    gnn_epochs: int = 10
    tt_batch_size: int = 256
    gnn_batch_size: int = 256


class ServeConfig(BaseModel):
    """Configuration for serving."""
    model_config = {"extra": "forbid"}

    host: str = "0.0.0.0"
    port: int = 8000
    top_k_feed: int = 50
    top_k_friends: int = 10
    cache_embeddings: bool = True


class Config(BaseModel):
    """Main configuration container."""
    model_config = {"extra": "forbid"}

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    @classmethod
    def small(cls) -> "Config":
        """Create config for small/quick testing."""
        return cls(
            data=SmallDataConfig(),
            train=SmallTrainConfig()
        )


def get_config(small: bool = False) -> Config:
    """Get configuration, optionally for small mode."""
    if small:
        return Config.small()
    return Config()
