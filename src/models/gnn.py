"""
Temporal Heterogeneous Graph Neural Network for social event recommendation.

Node types: user, event, venue
Edge types:
- user-user: friend
- user-event: view, like, save, intent, attend
- event-venue: hosted_at
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
import polars as pl
from typing import Optional
from datetime import datetime, timedelta


class TemporalHeteroGNN(nn.Module):
    """
    Heterogeneous Graph Transformer for calendar-aware social events.

    Uses HGTConv for message passing across different node and edge types.
    Incorporates temporal features and availability signals.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        # Feature dimensions
        user_feat_dim: int = 32,
        event_feat_dim: int = 32,
        venue_feat_dim: int = 16,
        # Number of entities for initial embeddings
        n_users: int = 5000,
        n_events: int = 30000,
        n_venues: int = 300,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Node type metadata
        self.node_types = ["user", "event", "venue"]
        self.edge_types = [
            ("user", "friend", "user"),
            ("user", "view", "event"),
            ("user", "like", "event"),
            ("user", "save", "event"),
            ("user", "intent", "event"),
            ("user", "attend", "event"),
            ("event", "rev_view", "user"),
            ("event", "rev_like", "user"),
            ("event", "rev_save", "user"),
            ("event", "rev_intent", "user"),
            ("event", "rev_attend", "user"),
            ("event", "hosted_at", "venue"),
            ("venue", "hosts", "event"),
        ]

        # Initial feature projections
        self.user_lin = Linear(user_feat_dim, hidden_dim)
        self.event_lin = Linear(event_feat_dim, hidden_dim)
        self.venue_lin = Linear(venue_feat_dim, hidden_dim)

        # Learnable initial embeddings (for cold start)
        self.user_embed = nn.Embedding(n_users, hidden_dim)
        self.event_embed = nn.Embedding(n_events, hidden_dim)
        self.venue_embed = nn.Embedding(n_venues, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(self.node_types, self.edge_types),
                heads=num_heads,
            )
            self.convs.append(conv)

        # Edge feature projections for user-event edges
        self.edge_feat_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),  # avail_fit, next_feasible, distance, friend_intent
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        node_ids_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through HGT layers.

        Args:
            x_dict: Dict mapping node type to feature tensor
            edge_index_dict: Dict mapping edge type to edge indices
            node_ids_dict: Optional dict mapping node type to original IDs
                          (for looking up learnable embeddings)

        Returns:
            Dict mapping node type to output embeddings
        """
        # Project input features to hidden dimension
        h_dict = {}

        # User features
        if "user" in x_dict and x_dict["user"] is not None:
            h_user = self.user_lin(x_dict["user"])
            # Add learnable embedding if IDs provided
            if node_ids_dict is not None and "user" in node_ids_dict:
                h_user = h_user + self.user_embed(node_ids_dict["user"])
            h_dict["user"] = h_user

        # Event features
        if "event" in x_dict and x_dict["event"] is not None:
            h_event = self.event_lin(x_dict["event"])
            if node_ids_dict is not None and "event" in node_ids_dict:
                h_event = h_event + self.event_embed(node_ids_dict["event"])
            h_dict["event"] = h_event

        # Venue features
        if "venue" in x_dict and x_dict["venue"] is not None:
            h_venue = self.venue_lin(x_dict["venue"])
            if node_ids_dict is not None and "venue" in node_ids_dict:
                h_venue = h_venue + self.venue_embed(node_ids_dict["venue"])
            h_dict["venue"] = h_venue

        # Apply HGT layers
        for conv in self.convs:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {key: self.dropout(F.relu(h)) for key, h in h_dict.items()}

        # Normalize outputs
        h_dict = {key: F.normalize(h, p=2, dim=-1) for key, h in h_dict.items()}

        return h_dict

    def compute_link_score(
        self,
        user_emb: torch.Tensor,
        event_emb: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute link prediction score for user-event pairs.

        Args:
            user_emb: (batch_size, hidden_dim)
            event_emb: (batch_size, hidden_dim)
            edge_features: (batch_size, 4) optional edge features

        Returns:
            scores: (batch_size,)
        """
        # Base dot product similarity
        score = (user_emb * event_emb).sum(dim=-1)

        # Add edge feature contribution if available
        if edge_features is not None:
            edge_h = self.edge_feat_proj(edge_features)
            edge_score = (user_emb * edge_h).sum(dim=-1)
            score = score + 0.1 * edge_score

        return score


class GNNLinkPrediction(nn.Module):
    """
    Wrapper for link prediction training with negative sampling.
    """

    def __init__(self, gnn: TemporalHeteroGNN):
        super().__init__()
        self.gnn = gnn

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        node_ids_dict: Optional[dict[str, torch.Tensor]] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for link prediction.

        Args:
            pos_edge_index: (2, n_pos) positive user-event edges
            neg_edge_index: (2, n_neg) negative user-event edges

        Returns:
            loss, pos_scores, neg_scores
        """
        # Get node embeddings
        h_dict = self.gnn(x_dict, edge_index_dict, node_ids_dict)

        # Get user and event embeddings for positive edges
        user_emb_pos = h_dict["user"][pos_edge_index[0]]
        event_emb_pos = h_dict["event"][pos_edge_index[1]]
        pos_scores = self.gnn.compute_link_score(user_emb_pos, event_emb_pos, edge_features)

        # Get embeddings for negative edges
        user_emb_neg = h_dict["user"][neg_edge_index[0]]
        event_emb_neg = h_dict["event"][neg_edge_index[1]]
        neg_scores = self.gnn.compute_link_score(user_emb_neg, event_emb_neg)

        # BPR loss
        loss = -F.logsigmoid(pos_scores - neg_scores.mean(dim=-1, keepdim=True).squeeze()).mean()

        return loss, pos_scores, neg_scores


def build_hetero_graph(
    users_df,
    events_df,
    venues_df,
    friendships_df,
    actions_df,
    cfg,
    cutoff_day: int = 10,
) -> HeteroData:
    """
    Build heterogeneous graph from dataframes.

    Args:
        cutoff_day: Only include actions up to this day (for temporal split)
    """
    data = HeteroData()

    # Build node features
    # User features: schedule embedding + interest vector
    n_users = len(users_df)
    user_features = np.zeros((n_users, 32), dtype=np.float32)

    interest_cols = [f"interest_{i}" for i in range(cfg.data.n_interest_dims)]
    user_interests = users_df.select(interest_cols).to_numpy()
    user_features[:, :cfg.data.n_interest_dims] = user_interests

    # Add schedule type as one-hot
    schedule_map = {"morning": 0, "night": 1, "flexible": 2}
    for i, stype in enumerate(users_df["schedule_type"].to_list()):
        user_features[i, cfg.data.n_interest_dims + schedule_map[stype]] = 1.0

    data["user"].x = torch.tensor(user_features)
    data["user"].num_nodes = n_users

    # Event features: time encoding + category
    n_events = len(events_df)
    event_features = np.zeros((n_events, 32), dtype=np.float32)

    for i, row in enumerate(events_df.iter_rows(named=True)):
        st = row["start_time"]
        hour = st.hour + st.minute / 60
        dow = st.weekday()

        # Time encoding
        event_features[i, 0] = np.sin(2 * np.pi * hour / 24)
        event_features[i, 1] = np.cos(2 * np.pi * hour / 24)
        event_features[i, 2] = np.sin(2 * np.pi * dow / 7)
        event_features[i, 3] = np.cos(2 * np.pi * dow / 7)

        # Normalized start time (day in horizon)
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        day_offset = (st - base_time).days
        event_features[i, 4] = day_offset / cfg.data.horizon_days

        # Duration normalized
        event_features[i, 5] = row["duration_min"] / 240

        # Category one-hot (truncated to fit)
        cat = row["category"]
        if cat < 20:  # Max categories we encode
            event_features[i, 10 + cat] = 1.0

    data["event"].x = torch.tensor(event_features)
    data["event"].num_nodes = n_events

    # Venue features: geo + popularity
    n_venues = len(venues_df)
    venue_features = np.zeros((n_venues, 16), dtype=np.float32)

    for i, row in enumerate(venues_df.iter_rows(named=True)):
        # Geo cell as normalized coordinates
        cell = row["geo_cell"]
        venue_features[i, 0] = (cell % 10) / 10
        venue_features[i, 1] = (cell // 10) / 10
        venue_features[i, 2] = np.log1p(row["popularity"]) / 5

    data["venue"].x = torch.tensor(venue_features)
    data["venue"].num_nodes = n_venues

    # Build edges
    # Friend edges (undirected)
    friend_src = friendships_df["user_id_1"].to_list()
    friend_dst = friendships_df["user_id_2"].to_list()
    # Make bidirectional
    friend_edge_index = torch.tensor([
        friend_src + friend_dst,
        friend_dst + friend_src
    ], dtype=torch.long)
    data["user", "friend", "user"].edge_index = friend_edge_index

    # Filter actions by cutoff
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_time = base_time + timedelta(days=cutoff_day)

    actions_filtered = actions_df.filter(pl.col("ts") < cutoff_time)

    # User-event edges by action type
    for action_type in ["view", "like", "save", "intent", "attend"]:
        action_edges = actions_filtered.filter(pl.col("action_type") == action_type)
        if len(action_edges) > 0:
            src = action_edges["user_id"].to_list()
            dst = action_edges["event_id"].to_list()
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            data["user", action_type, "event"].edge_index = edge_index
            # Reverse edges
            data["event", f"rev_{action_type}", "user"].edge_index = torch.tensor([dst, src], dtype=torch.long)

    # Event-venue edges
    event_venues = events_df.select(["event_id", "venue_id"]).to_numpy()
    data["event", "hosted_at", "venue"].edge_index = torch.tensor(
        event_venues.T, dtype=torch.long
    )
    data["venue", "hosts", "event"].edge_index = torch.tensor(
        event_venues[:, ::-1].T.copy(), dtype=torch.long
    )

    return data


def extract_gnn_embeddings(
    model: TemporalHeteroGNN,
    data: HeteroData,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract GNN embeddings for all users and events.

    Returns:
        user_embeddings: (n_users, hidden_dim)
        event_embeddings: (n_events, hidden_dim)
    """
    model.eval()

    with torch.no_grad():
        # Move data to device
        x_dict = {
            "user": data["user"].x.to(device),
            "event": data["event"].x.to(device),
            "venue": data["venue"].x.to(device),
        }

        edge_index_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], "edge_index"):
                edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)

        # Forward pass
        h_dict = model(x_dict, edge_index_dict)

        user_emb = h_dict["user"].cpu()
        event_emb = h_dict["event"].cpu()

    return user_emb, event_emb


