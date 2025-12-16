"""
Smoke tests for the social event recommender.
These tests verify basic functionality without requiring full training.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        from src.config import get_config
        cfg = get_config(small=True)
        assert cfg.data.n_users == 500
        assert cfg.data.n_events == 3000

    def test_config_full(self):
        from src.config import get_config
        cfg = get_config(small=False)
        assert cfg.data.n_users == 5000
        assert cfg.data.n_events == 30000


class TestDataGeneration:
    """Test synthetic data generation."""

    def test_generate_users(self):
        from src.config import get_config, DataConfig
        from src.data.gen_synth import generate_users, set_seed

        cfg = DataConfig(n_users=100, n_communities=3)
        set_seed(42)
        users = generate_users(cfg)

        assert len(users) == 100
        assert "user_id" in users.columns
        assert "home_geo_cell" in users.columns
        assert "interest_0" in users.columns

    def test_generate_friendships(self):
        from src.config import DataConfig
        from src.data.gen_synth import generate_users, generate_friendships, set_seed

        cfg = DataConfig(n_users=100, n_communities=3)
        set_seed(42)
        users = generate_users(cfg)
        friendships = generate_friendships(cfg, users)

        assert len(friendships) > 0
        assert "user_id_1" in friendships.columns
        assert "user_id_2" in friendships.columns

    def test_generate_venues(self):
        from src.config import DataConfig
        from src.data.gen_synth import generate_venues, set_seed

        cfg = DataConfig(n_venues=50)
        set_seed(42)
        venues = generate_venues(cfg)

        assert len(venues) == 50
        assert "venue_id" in venues.columns
        assert "geo_cell" in venues.columns


class TestModels:
    """Test model architectures."""

    def test_two_tower_forward(self):
        from src.models.two_tower import TwoTowerModel

        model = TwoTowerModel(
            n_users=100,
            n_events=500,
            n_venues=50,
            n_categories=10,
            n_geo_cells=100,
            embed_dim=32,
        )

        # Create dummy batch
        batch_size = 4
        user_features = {
            "user_id": torch.randint(0, 100, (batch_size,)),
            "geo_cell": torch.randint(0, 100, (batch_size,)),
            "age_bucket": torch.randint(0, 5, (batch_size,)),
            "schedule_type": torch.randint(0, 3, (batch_size,)),
            "interest_vector": torch.randn(batch_size, 16),
        }

        event_features = {
            "event_id": torch.randint(0, 500, (batch_size,)),
            "venue_id": torch.randint(0, 50, (batch_size,)),
            "category": torch.randint(0, 10, (batch_size,)),
            "geo_cell": torch.randint(0, 100, (batch_size,)),
            "time_features": torch.randn(batch_size, 4),
            "duration_norm": torch.rand(batch_size),
            "capacity_norm": torch.rand(batch_size),
            "freshness_norm": torch.rand(batch_size),
        }

        user_emb, event_emb = model(user_features, event_features)

        assert user_emb.shape == (batch_size, 32)
        assert event_emb.shape == (batch_size, 32)

    def test_reranker_forward(self):
        from src.models.reranker import RerankerMLP

        model = RerankerMLP(
            n_engineered_features=7,
            hidden_dims=[32, 16],
            use_gnn_embeddings=False,
        )

        batch_size = 4
        tt_sim = torch.randn(batch_size)
        features = torch.randn(batch_size, 7)

        scores = model(tt_sim, features)
        assert scores.shape == (batch_size,)

    def test_gnn_forward(self):
        pytest.importorskip("torch_geometric")
        from src.models.gnn import TemporalHeteroGNN

        model = TemporalHeteroGNN(
            hidden_dim=32,
            num_layers=2,
            num_heads=2,
            n_users=100,
            n_events=500,
            n_venues=50,
        )

        # Create minimal graph data
        x_dict = {
            "user": torch.randn(100, 32),
            "event": torch.randn(500, 32),
            "venue": torch.randn(50, 16),
        }

        edge_index_dict = {
            ("user", "friend", "user"): torch.randint(0, 100, (2, 50)),
            ("user", "intent", "event"): torch.stack([
                torch.randint(0, 100, (30,)),
                torch.randint(0, 500, (30,))
            ]),
            ("event", "rev_intent", "user"): torch.stack([
                torch.randint(0, 500, (30,)),
                torch.randint(0, 100, (30,))
            ]),
            ("event", "hosted_at", "venue"): torch.stack([
                torch.arange(500),
                torch.randint(0, 50, (500,))
            ]),
            ("venue", "hosts", "event"): torch.stack([
                torch.randint(0, 50, (500,)),
                torch.arange(500)
            ]),
        }

        h_dict = model(x_dict, edge_index_dict)

        assert "user" in h_dict
        assert "event" in h_dict
        assert "venue" in h_dict
        assert h_dict["user"].shape == (100, 32)


class TestAvailability:
    """Test availability computation."""

    def test_geo_distance(self):
        from src.data.gen_synth import compute_geo_distance

        # Same cell
        assert compute_geo_distance(0, 0) == 0.0

        # Adjacent cells
        dist = compute_geo_distance(0, 1)
        assert dist > 0

        # Diagonal cells
        dist_diag = compute_geo_distance(0, 11)  # 10x10 grid
        assert dist_diag > dist


class TestIntegration:
    """Integration tests (require generated data)."""

    @pytest.mark.skipif(
        not Path("data/users.parquet").exists(),
        reason="Data not generated yet"
    )
    def test_load_data(self):
        import polars as pl

        users = pl.read_parquet("data/users.parquet")
        assert len(users) > 0

        events = pl.read_parquet("data/events.parquet")
        assert len(events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
