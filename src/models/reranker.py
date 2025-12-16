"""
MLP Reranker model that combines two-tower similarity with engineered features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class RerankerMLP(nn.Module):
    """
    MLP reranker that takes:
    - Two-tower similarity score
    - Engineered features (availability, distance, friends, etc.)
    - Optional: GNN-refined embeddings

    Outputs a final ranking score.
    """

    def __init__(
        self,
        n_engineered_features: int = 7,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_gnn_embeddings: bool = False,
        gnn_embed_dim: int = 64,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Input: tt_score + engineered_features + optional gnn
        input_dim = 1 + n_engineered_features
        if use_gnn_embeddings:
            input_dim += gnn_embed_dim * 2  # user + event GNN embeddings

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self.use_gnn_embeddings = use_gnn_embeddings

        # For feature importance tracking
        self.feature_names = [
            "tt_similarity",
            "availability_fit",
            "next_feasible_hours",
            "distance_km",
            "friend_intent_count",
            "friend_feasible_count",
            "freshness_hours",
            "category_match",
        ]

    def forward(
        self,
        tt_similarity: torch.Tensor,
        engineered_features: torch.Tensor,
        gnn_user_emb: Optional[torch.Tensor] = None,
        gnn_event_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tt_similarity: (batch_size,) two-tower similarity scores
            engineered_features: (batch_size, n_features) engineered features
            gnn_user_emb: (batch_size, gnn_embed_dim) optional GNN user embeddings
            gnn_event_emb: (batch_size, gnn_embed_dim) optional GNN event embeddings

        Returns:
            scores: (batch_size,) ranking scores
        """
        # Concatenate inputs
        inputs = [tt_similarity.unsqueeze(-1), engineered_features]

        if self.use_gnn_embeddings and gnn_user_emb is not None and gnn_event_emb is not None:
            inputs.extend([gnn_user_emb, gnn_event_emb])

        x = torch.cat(inputs, dim=-1)
        return self.net(x).squeeze(-1)


class RerankerWithExplanation(RerankerMLP):
    """
    Reranker that provides feature importance explanations.
    Uses gradient-based saliency for feature attribution.
    """

    def get_feature_importance(
        self,
        tt_similarity: torch.Tensor,
        engineered_features: torch.Tensor,
        gnn_user_emb: Optional[torch.Tensor] = None,
        gnn_event_emb: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        Compute feature importance via input gradients.

        Returns dict mapping feature name to importance score.
        """
        # Enable gradients
        tt_similarity = tt_similarity.clone().requires_grad_(True)
        engineered_features = engineered_features.clone().requires_grad_(True)

        # Forward pass
        score = self.forward(
            tt_similarity,
            engineered_features,
            gnn_user_emb,
            gnn_event_emb
        )

        # Backward to get gradients
        score.sum().backward()

        # Collect importances
        importances = {}

        # TT similarity importance
        if tt_similarity.grad is not None:
            importances["tt_similarity"] = float(
                (tt_similarity.grad.abs() * tt_similarity.abs()).mean()
            )

        # Engineered feature importances
        if engineered_features.grad is not None:
            for i, name in enumerate(self.feature_names[1:]):  # Skip tt_similarity
                if i < engineered_features.shape[-1]:
                    grad = engineered_features.grad[..., i]
                    feat = engineered_features[..., i]
                    importances[name] = float((grad.abs() * feat.abs()).mean())

        # Normalize to sum to 1
        total = sum(importances.values()) + 1e-8
        importances = {k: v / total for k, v in importances.items()}

        return importances


def prepare_reranker_features(
    candidates_df,
    normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels from candidates dataframe.

    Returns:
        features: (n_samples, n_features)
        labels: (n_samples,) binary intent labels
    """
    feature_cols = [
        "availability_fit",
        "next_feasible_hours",
        "distance_km",
        "friend_intent_count",
        "friend_feasible_count",
        "freshness_hours",
        "category_match",
    ]

    features = candidates_df.select(feature_cols).to_numpy()

    if normalize:
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize each feature
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0) + 1e-8
        features = (features - means) / stds

    labels = candidates_df["intent"].to_numpy()

    return features.astype(np.float32), labels.astype(np.float32)


class RerankerDataset(torch.utils.data.Dataset):
    """Dataset for reranker training."""

    def __init__(
        self,
        tt_similarities: np.ndarray,
        engineered_features: np.ndarray,
        labels: np.ndarray,
        gnn_user_embs: Optional[np.ndarray] = None,
        gnn_event_embs: Optional[np.ndarray] = None,
    ):
        self.tt_similarities = torch.tensor(tt_similarities, dtype=torch.float32)
        self.engineered_features = torch.tensor(engineered_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.gnn_user_embs = None
        self.gnn_event_embs = None
        if gnn_user_embs is not None:
            self.gnn_user_embs = torch.tensor(gnn_user_embs, dtype=torch.float32)
        if gnn_event_embs is not None:
            self.gnn_event_embs = torch.tensor(gnn_event_embs, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "tt_similarity": self.tt_similarities[idx],
            "engineered_features": self.engineered_features[idx],
            "label": self.labels[idx],
        }

        if self.gnn_user_embs is not None:
            item["gnn_user_emb"] = self.gnn_user_embs[idx]
        if self.gnn_event_embs is not None:
            item["gnn_event_emb"] = self.gnn_event_embs[idx]

        return item
