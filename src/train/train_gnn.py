"""
Training script for temporal heterogeneous GNN.
"""
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.models.gnn import TemporalHeteroGNN, build_hetero_graph, extract_gnn_embeddings
from src.models.reranker import RerankerMLP, RerankerDataset, prepare_reranker_features


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinkPredictionDataset(Dataset):
    """Dataset for GNN link prediction training."""

    def __init__(
        self,
        positive_edges: np.ndarray,  # (n_pos, 2) user_id, event_id pairs
        n_users: int,
        n_events: int,
        n_neg: int = 5,
    ):
        self.positive_edges = positive_edges
        self.n_users = n_users
        self.n_events = n_events
        self.n_neg = n_neg

    def __len__(self) -> int:
        return len(self.positive_edges)

    def __getitem__(self, idx: int) -> dict:
        user_id, event_id = self.positive_edges[idx]

        # Sample negative events
        neg_events = np.random.choice(
            [e for e in range(self.n_events) if e != event_id],
            size=self.n_neg,
            replace=False
        )

        return {
            "user_id": int(user_id),
            "pos_event_id": int(event_id),
            "neg_event_ids": neg_events.astype(np.int64),
        }


def train_gnn_epoch(
    model: TemporalHeteroGNN,
    data,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train GNN for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    # Move graph data to device
    x_dict = {
        "user": data["user"].x.to(device),
        "event": data["event"].x.to(device),
        "venue": data["venue"].x.to(device),
    }

    edge_index_dict = {}
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], "edge_index"):
            edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)

    for batch in train_loader:
        optimizer.zero_grad()

        # Get node embeddings from GNN
        h_dict = model(x_dict, edge_index_dict)

        user_ids = batch["user_id"].to(device)
        pos_event_ids = batch["pos_event_id"].to(device)
        neg_event_ids = batch["neg_event_ids"].to(device)  # (batch_size, n_neg)

        # Get embeddings for positive pairs
        user_emb = h_dict["user"][user_ids]
        pos_event_emb = h_dict["event"][pos_event_ids]
        pos_scores = (user_emb * pos_event_emb).sum(dim=-1)

        # Get embeddings for negative pairs
        batch_size, n_neg = neg_event_ids.shape
        neg_event_emb = h_dict["event"][neg_event_ids.flatten()].reshape(batch_size, n_neg, -1)
        neg_scores = torch.bmm(neg_event_emb, user_emb.unsqueeze(-1)).squeeze(-1)

        # BPR loss
        diff = pos_scores.unsqueeze(-1) - neg_scores
        loss = -F.logsigmoid(diff).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_gnn(
    model: TemporalHeteroGNN,
    data,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate GNN on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    total_hits = 0
    total_samples = 0

    x_dict = {
        "user": data["user"].x.to(device),
        "event": data["event"].x.to(device),
        "venue": data["venue"].x.to(device),
    }

    edge_index_dict = {}
    for edge_type in data.edge_types:
        if hasattr(data[edge_type], "edge_index"):
            edge_index_dict[edge_type] = data[edge_type].edge_index.to(device)

    with torch.no_grad():
        h_dict = model(x_dict, edge_index_dict)

        for batch in val_loader:
            user_ids = batch["user_id"].to(device)
            pos_event_ids = batch["pos_event_id"].to(device)
            neg_event_ids = batch["neg_event_ids"].to(device)

            user_emb = h_dict["user"][user_ids]
            pos_event_emb = h_dict["event"][pos_event_ids]
            pos_scores = (user_emb * pos_event_emb).sum(dim=-1)

            batch_size, n_neg = neg_event_ids.shape
            neg_event_emb = h_dict["event"][neg_event_ids.flatten()].reshape(batch_size, n_neg, -1)
            neg_scores = torch.bmm(neg_event_emb, user_emb.unsqueeze(-1)).squeeze(-1)

            diff = pos_scores.unsqueeze(-1) - neg_scores
            loss = -F.logsigmoid(diff).mean()

            total_loss += loss.item()
            n_batches += 1

            # Hit rate: positive ranked above all negatives
            hits = (pos_scores.unsqueeze(-1) > neg_scores).all(dim=-1).float().sum()
            total_hits += hits.item()
            total_samples += batch_size

    return {
        "val_loss": total_loss / n_batches,
        "hit_rate": total_hits / total_samples,
    }


def train_reranker(
    candidates_df: pl.DataFrame,
    user_embeddings: torch.Tensor,
    event_embeddings: torch.Tensor,
    gnn_user_embs: torch.Tensor,
    gnn_event_embs: torch.Tensor,
    cfg,
    device: str,
) -> RerankerMLP:
    """Train the fusion reranker with GNN embeddings."""
    logger.info("Training reranker...")

    # Prepare features
    features, labels = prepare_reranker_features(candidates_df)

    # Compute two-tower similarities
    user_ids = candidates_df["user_id"].to_numpy()
    event_ids = candidates_df["event_id"].to_numpy()

    tt_user_emb = user_embeddings[user_ids].numpy()
    tt_event_emb = event_embeddings[event_ids].numpy()
    tt_similarities = (tt_user_emb * tt_event_emb).sum(axis=1)

    # Get GNN embeddings for candidates
    gnn_user = gnn_user_embs[user_ids].numpy()
    gnn_event = gnn_event_embs[event_ids].numpy()

    # Create dataset
    dataset = RerankerDataset(
        tt_similarities=tt_similarities,
        engineered_features=features,
        labels=labels,
        gnn_user_embs=gnn_user,
        gnn_event_embs=gnn_event,
    )

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.rr_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.rr_batch_size, shuffle=False)

    # Create model
    model = RerankerMLP(
        n_engineered_features=features.shape[1],
        hidden_dims=cfg.model.reranker_hidden_dims,
        dropout=cfg.model.reranker_dropout,
        use_gnn_embeddings=True,
        gnn_embed_dim=cfg.model.gnn_hidden_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.rr_lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(cfg.train.rr_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            scores = model(
                batch["tt_similarity"].to(device),
                batch["engineered_features"].to(device),
                batch["gnn_user_emb"].to(device),
                batch["gnn_event_emb"].to(device),
            )

            loss = criterion(scores, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                scores = model(
                    batch["tt_similarity"].to(device),
                    batch["engineered_features"].to(device),
                    batch["gnn_user_emb"].to(device),
                    batch["gnn_event_emb"].to(device),
                )
                loss = criterion(scores, batch["label"].to(device))
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), cfg.train.models_dir / "reranker_best.pt")

        logger.info(f"Reranker Epoch {epoch + 1}/{cfg.train.rr_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train GNN model")
    parser.add_argument("--small", action="store_true", help="Use small config")
    args = parser.parse_args()

    cfg = get_config(small=args.small)
    set_seed(cfg.train.seed)

    device = cfg.train.device
    logger.info(f"Using device: {device}")

    data_dir = cfg.data.data_dir
    models_dir = cfg.train.models_dir
    runs_dir = cfg.train.runs_dir

    models_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    users = pl.read_parquet(data_dir / "users.parquet")
    events = pl.read_parquet(data_dir / "events.parquet")
    venues = pl.read_parquet(data_dir / "venues.parquet")
    friendships = pl.read_parquet(data_dir / "friendships.parquet")
    actions = pl.read_parquet(data_dir / "actions.parquet")

    # Build heterogeneous graph (with train cutoff)
    logger.info("Building heterogeneous graph...")
    train_cutoff = cfg.train.train_days[1]
    data = build_hetero_graph(users, events, venues, friendships, actions, cfg, cutoff_day=train_cutoff)

    logger.info(f"Graph nodes: user={data['user'].num_nodes}, event={data['event'].num_nodes}, venue={data['venue'].num_nodes}")
    logger.info(f"Graph edge types: {data.edge_types}")

    # Get positive edges for training (intent actions)
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    train_end = base_time + timedelta(days=cfg.train.train_days[1])
    val_start = base_time + timedelta(days=cfg.train.val_days[0] - 1)
    val_end = base_time + timedelta(days=cfg.train.val_days[1])

    intent_actions = actions.filter(pl.col("action_type") == "intent")
    train_intents = intent_actions.filter(pl.col("ts") < train_end)
    val_intents = intent_actions.filter(
        (pl.col("ts") >= val_start) & (pl.col("ts") < val_end)
    )

    train_edges = train_intents.select(["user_id", "event_id"]).to_numpy()
    val_edges = val_intents.select(["user_id", "event_id"]).to_numpy()

    logger.info(f"Train edges: {len(train_edges)}, Val edges: {len(val_edges)}")

    # Create datasets
    train_dataset = LinkPredictionDataset(
        train_edges,
        n_users=len(users),
        n_events=len(events),
        n_neg=cfg.train.gnn_neg_samples,
    )

    val_dataset = LinkPredictionDataset(
        val_edges,
        n_users=len(users),
        n_events=len(events),
        n_neg=cfg.train.gnn_neg_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.gnn_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.gnn_batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create GNN model
    logger.info("Creating GNN model...")
    model = TemporalHeteroGNN(
        hidden_dim=cfg.model.gnn_hidden_dim,
        num_layers=cfg.model.gnn_num_layers,
        num_heads=cfg.model.gnn_num_heads,
        dropout=cfg.model.gnn_dropout,
        n_users=len(users),
        n_events=len(events),
        n_venues=len(venues),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.train.gnn_lr)

    # Training loop
    logger.info("Starting GNN training...")
    best_val_loss = float("inf")
    metrics_history = []

    for epoch in range(cfg.train.gnn_epochs):
        train_loss = train_gnn_epoch(model, data, train_loader, optimizer, device)
        val_metrics = evaluate_gnn(model, data, val_loader, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        metrics_history.append(metrics)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.train.gnn_epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - "
            f"Hit Rate: {val_metrics['hit_rate']:.4f}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), models_dir / "gnn_best.pt")

    # Save final model
    torch.save(model.state_dict(), models_dir / "gnn_final.pt")

    # Extract GNN embeddings
    logger.info("Extracting GNN embeddings...")
    model.load_state_dict(torch.load(models_dir / "gnn_best.pt"))
    gnn_user_embs, gnn_event_embs = extract_gnn_embeddings(model, data, device)

    torch.save(gnn_user_embs, models_dir / "gnn_user_embeddings.pt")
    torch.save(gnn_event_embs, models_dir / "gnn_event_embeddings.pt")

    logger.info(f"Saved GNN user embeddings: {gnn_user_embs.shape}")
    logger.info(f"Saved GNN event embeddings: {gnn_event_embs.shape}")

    # Load two-tower embeddings
    user_embeddings = torch.load(models_dir / "user_embeddings.pt")
    event_embeddings = torch.load(models_dir / "event_embeddings.pt")

    # Train reranker
    train_candidates = pl.read_parquet(data_dir / "train_candidates.parquet")
    reranker = train_reranker(
        train_candidates,
        user_embeddings,
        event_embeddings,
        gnn_user_embs,
        gnn_event_embs,
        cfg,
        device,
    )

    # Save reranker
    torch.save(reranker.state_dict(), models_dir / "reranker_final.pt")

    # Save metrics
    with open(runs_dir / "gnn_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("GNN training complete!")


if __name__ == "__main__":
    main()
