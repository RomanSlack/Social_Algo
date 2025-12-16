"""
Two-tower retrieval model for candidate generation.
User tower and event tower learn embeddings, scored via dot product.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UserTower(nn.Module):
    """User tower: encodes user features into embedding."""

    def __init__(
        self,
        n_users: int,
        n_geo_cells: int,
        n_age_buckets: int,
        n_schedule_types: int,
        interest_dim: int,
        embed_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()

        # Embedding layers
        self.user_embed = nn.Embedding(n_users, embed_dim // 2)
        self.geo_embed = nn.Embedding(n_geo_cells, 8)
        self.age_embed = nn.Embedding(n_age_buckets, 4)
        self.schedule_embed = nn.Embedding(n_schedule_types, 4)

        # MLP to combine features
        mlp_input_dim = embed_dim // 2 + 8 + 4 + 4 + interest_dim
        self.mlp = MLP(mlp_input_dim, hidden_dims, embed_dim)

    def forward(
        self,
        user_ids: torch.Tensor,
        geo_cells: torch.Tensor,
        age_buckets: torch.Tensor,
        schedule_types: torch.Tensor,
        interest_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            user_ids: (batch_size,)
            geo_cells: (batch_size,)
            age_buckets: (batch_size,)
            schedule_types: (batch_size,)
            interest_vectors: (batch_size, interest_dim)

        Returns:
            user_embeddings: (batch_size, embed_dim)
        """
        user_emb = self.user_embed(user_ids)
        geo_emb = self.geo_embed(geo_cells)
        age_emb = self.age_embed(age_buckets)
        schedule_emb = self.schedule_embed(schedule_types)

        # Concatenate all features
        x = torch.cat([user_emb, geo_emb, age_emb, schedule_emb, interest_vectors], dim=-1)

        # Project to final embedding
        output = self.mlp(x)
        return F.normalize(output, p=2, dim=-1)


class EventTower(nn.Module):
    """Event tower: encodes event features into embedding."""

    def __init__(
        self,
        n_events: int,
        n_venues: int,
        n_categories: int,
        n_geo_cells: int,
        embed_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()

        # Embedding layers
        self.event_embed = nn.Embedding(n_events, embed_dim // 2)
        self.venue_embed = nn.Embedding(n_venues, 16)
        self.category_embed = nn.Embedding(n_categories, 8)
        self.geo_embed = nn.Embedding(n_geo_cells, 8)

        # Time features (continuous)
        self.time_proj = nn.Linear(4, 8)  # hour_sin, hour_cos, dow_sin, dow_cos

        # MLP to combine
        mlp_input_dim = embed_dim // 2 + 16 + 8 + 8 + 8 + 3  # +3 for duration, capacity, freshness
        self.mlp = MLP(mlp_input_dim, hidden_dims, embed_dim)

    def forward(
        self,
        event_ids: torch.Tensor,
        venue_ids: torch.Tensor,
        categories: torch.Tensor,
        geo_cells: torch.Tensor,
        time_features: torch.Tensor,  # (batch, 4): hour_sin, hour_cos, dow_sin, dow_cos
        duration_norm: torch.Tensor,
        capacity_norm: torch.Tensor,
        freshness_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            event_embeddings: (batch_size, embed_dim)
        """
        event_emb = self.event_embed(event_ids)
        venue_emb = self.venue_embed(venue_ids)
        cat_emb = self.category_embed(categories)
        geo_emb = self.geo_embed(geo_cells)
        time_emb = self.time_proj(time_features)

        # Concatenate all features
        x = torch.cat([
            event_emb, venue_emb, cat_emb, geo_emb, time_emb,
            duration_norm.unsqueeze(-1),
            capacity_norm.unsqueeze(-1),
            freshness_norm.unsqueeze(-1)
        ], dim=-1)

        output = self.mlp(x)
        return F.normalize(output, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-tower retrieval model.
    Trains with sampled softmax / BPR loss.
    """

    def __init__(
        self,
        n_users: int,
        n_events: int,
        n_venues: int,
        n_categories: int,
        n_geo_cells: int,
        n_age_buckets: int = 5,
        n_schedule_types: int = 3,
        interest_dim: int = 16,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.user_tower = UserTower(
            n_users=n_users,
            n_geo_cells=n_geo_cells,
            n_age_buckets=n_age_buckets,
            n_schedule_types=n_schedule_types,
            interest_dim=interest_dim,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
        )

        self.event_tower = EventTower(
            n_events=n_events,
            n_venues=n_venues,
            n_categories=n_categories,
            n_geo_cells=n_geo_cells,
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
        )

        self.embed_dim = embed_dim

    def forward(
        self,
        user_features: dict[str, torch.Tensor],
        event_features: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute user and event embeddings.

        Returns:
            user_emb: (batch_size, embed_dim)
            event_emb: (batch_size, embed_dim)
        """
        user_emb = self.user_tower(
            user_ids=user_features["user_id"],
            geo_cells=user_features["geo_cell"],
            age_buckets=user_features["age_bucket"],
            schedule_types=user_features["schedule_type"],
            interest_vectors=user_features["interest_vector"],
        )

        event_emb = self.event_tower(
            event_ids=event_features["event_id"],
            venue_ids=event_features["venue_id"],
            categories=event_features["category"],
            geo_cells=event_features["geo_cell"],
            time_features=event_features["time_features"],
            duration_norm=event_features["duration_norm"],
            capacity_norm=event_features["capacity_norm"],
            freshness_norm=event_features["freshness_norm"],
        )

        return user_emb, event_emb

    def compute_score(
        self,
        user_emb: torch.Tensor,
        event_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot product similarity scores."""
        return (user_emb * event_emb).sum(dim=-1)

    def compute_bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_event_emb: torch.Tensor,
        neg_event_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BPR (Bayesian Personalized Ranking) loss.

        Args:
            user_emb: (batch_size, embed_dim)
            pos_event_emb: (batch_size, embed_dim)
            neg_event_embs: (batch_size, n_neg, embed_dim)

        Returns:
            loss: scalar
        """
        # Positive scores
        pos_scores = (user_emb * pos_event_emb).sum(dim=-1)  # (batch_size,)

        # Negative scores
        neg_scores = torch.bmm(
            neg_event_embs,
            user_emb.unsqueeze(-1)
        ).squeeze(-1)  # (batch_size, n_neg)

        # BPR loss: -log(sigmoid(pos - neg))
        diff = pos_scores.unsqueeze(-1) - neg_scores  # (batch_size, n_neg)
        loss = -F.logsigmoid(diff).mean()

        return loss


def extract_all_user_embeddings(
    model: TwoTowerModel,
    users_df,
    interest_cols: list[str],
    device: str = "cuda",
    batch_size: int = 1024
) -> torch.Tensor:
    """Extract embeddings for all users."""
    import numpy as np

    model.eval()
    n_users = len(users_df)
    embeddings = []

    # Map categorical features
    age_map = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
    schedule_map = {"morning": 0, "night": 1, "flexible": 2}

    with torch.no_grad():
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch = users_df[start:end]

            user_ids = torch.tensor(batch["user_id"].to_list(), device=device)
            geo_cells = torch.tensor(batch["home_geo_cell"].to_list(), device=device)
            age_buckets = torch.tensor(
                [age_map[a] for a in batch["age_bucket"].to_list()],
                device=device
            )
            schedule_types = torch.tensor(
                [schedule_map[s] for s in batch["schedule_type"].to_list()],
                device=device
            )

            interest_vectors = torch.tensor(
                batch.select(interest_cols).to_numpy(),
                dtype=torch.float32,
                device=device
            )

            user_features = {
                "user_id": user_ids,
                "geo_cell": geo_cells,
                "age_bucket": age_buckets,
                "schedule_type": schedule_types,
                "interest_vector": interest_vectors,
            }

            user_emb = model.user_tower(
                user_ids=user_features["user_id"],
                geo_cells=user_features["geo_cell"],
                age_buckets=user_features["age_bucket"],
                schedule_types=user_features["schedule_type"],
                interest_vectors=user_features["interest_vector"],
            )

            embeddings.append(user_emb.cpu())

    return torch.cat(embeddings, dim=0)


def extract_all_event_embeddings(
    model: TwoTowerModel,
    events_df,
    venues_df,
    device: str = "cuda",
    batch_size: int = 1024
) -> torch.Tensor:
    """Extract embeddings for all events."""
    import numpy as np
    from datetime import datetime

    model.eval()
    n_events = len(events_df)
    embeddings = []

    # Build venue geo lookup
    venue_geo = dict(zip(venues_df["venue_id"].to_list(), venues_df["geo_cell"].to_list()))

    # Normalization constants
    max_duration = 240
    max_capacity = 1000
    max_freshness = 7 * 24  # hours

    with torch.no_grad():
        for start in range(0, n_events, batch_size):
            end = min(start + batch_size, n_events)
            batch = events_df[start:end]

            event_ids = torch.tensor(batch["event_id"].to_list(), device=device)
            venue_ids = torch.tensor(batch["venue_id"].to_list(), device=device)
            categories = torch.tensor(batch["category"].to_list(), device=device)

            geo_cells = torch.tensor(
                [venue_geo.get(v, 0) for v in batch["venue_id"].to_list()],
                device=device
            )

            # Time features
            time_features_list = []
            for st in batch["start_time"].to_list():
                hour = st.hour + st.minute / 60
                dow = st.weekday()
                time_features_list.append([
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7),
                    np.cos(2 * np.pi * dow / 7),
                ])
            time_features = torch.tensor(time_features_list, dtype=torch.float32, device=device)

            duration_norm = torch.tensor(
                [d / max_duration for d in batch["duration_min"].to_list()],
                dtype=torch.float32,
                device=device
            )

            capacity_norm = torch.tensor(
                [np.log1p(c) / np.log1p(max_capacity) for c in batch["capacity"].to_list()],
                dtype=torch.float32,
                device=device
            )

            # Freshness (hours since posting)
            freshness_list = []
            for st, ft in zip(batch["start_time"].to_list(), batch["freshness_ts"].to_list()):
                hours = (st - ft).total_seconds() / 3600
                freshness_list.append(min(hours, max_freshness) / max_freshness)
            freshness_norm = torch.tensor(freshness_list, dtype=torch.float32, device=device)

            event_features = {
                "event_id": event_ids,
                "venue_id": venue_ids,
                "category": categories,
                "geo_cell": geo_cells,
                "time_features": time_features,
                "duration_norm": duration_norm,
                "capacity_norm": capacity_norm,
                "freshness_norm": freshness_norm,
            }

            event_emb = model.event_tower(
                event_ids=event_features["event_id"],
                venue_ids=event_features["venue_id"],
                categories=event_features["category"],
                geo_cells=event_features["geo_cell"],
                time_features=event_features["time_features"],
                duration_norm=event_features["duration_norm"],
                capacity_norm=event_features["capacity_norm"],
                freshness_norm=event_features["freshness_norm"],
            )

            embeddings.append(event_emb.cpu())

    return torch.cat(embeddings, dim=0)
