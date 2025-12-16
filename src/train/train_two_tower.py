"""
Training script for two-tower retrieval model.
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
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.models.two_tower import TwoTowerModel, extract_all_user_embeddings, extract_all_event_embeddings


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwoTowerDataset(Dataset):
    """Dataset for two-tower training with positive engagement sampling."""

    def __init__(
        self,
        actions_df: pl.DataFrame,
        users_df: pl.DataFrame,
        events_df: pl.DataFrame,
        venues_df: pl.DataFrame,
        cfg,
        target_action: str = "intent",
    ):
        self.cfg = cfg

        # Filter to target action
        positive_df = actions_df.filter(pl.col("action_type") == target_action)

        self.user_ids = positive_df["user_id"].to_numpy()
        self.event_ids = positive_df["event_id"].to_numpy()

        # Build user feature lookup
        self.user_features = self._build_user_features(users_df)

        # Build event feature lookup
        self.event_features = self._build_event_features(events_df, venues_df)

        # All events for negative sampling
        self.all_event_ids = events_df["event_id"].to_numpy()

        logger.info(f"Created dataset with {len(self)} positive pairs")

    def _build_user_features(self, users_df: pl.DataFrame) -> dict:
        """Build user feature lookup dict."""
        age_map = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
        schedule_map = {"morning": 0, "night": 1, "flexible": 2}

        interest_cols = [f"interest_{i}" for i in range(self.cfg.data.n_interest_dims)]

        features = {}
        for row in users_df.iter_rows(named=True):
            user_id = row["user_id"]
            features[user_id] = {
                "geo_cell": row["home_geo_cell"],
                "age_bucket": age_map[row["age_bucket"]],
                "schedule_type": schedule_map[row["schedule_type"]],
                "interest_vector": np.array([row[c] for c in interest_cols], dtype=np.float32),
            }
        return features

    def _build_event_features(self, events_df: pl.DataFrame, venues_df: pl.DataFrame) -> dict:
        """Build event feature lookup dict."""
        venue_geo = dict(zip(venues_df["venue_id"].to_list(), venues_df["geo_cell"].to_list()))

        features = {}
        base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for row in events_df.iter_rows(named=True):
            event_id = row["event_id"]
            st = row["start_time"]
            hour = st.hour + st.minute / 60
            dow = st.weekday()

            features[event_id] = {
                "venue_id": row["venue_id"],
                "category": row["category"],
                "geo_cell": venue_geo.get(row["venue_id"], 0),
                "time_features": np.array([
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * dow / 7),
                    np.cos(2 * np.pi * dow / 7),
                ], dtype=np.float32),
                "duration_norm": row["duration_min"] / 240,
                "capacity_norm": np.log1p(row["capacity"]) / np.log1p(1000),
                "freshness_norm": min((st - row["freshness_ts"]).total_seconds() / 3600 / 168, 1.0),
            }
        return features

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> dict:
        user_id = int(self.user_ids[idx])
        pos_event_id = int(self.event_ids[idx])

        # Get user features
        uf = self.user_features[user_id]

        # Get positive event features
        pf = self.event_features[pos_event_id]

        return {
            "user_id": user_id,
            "user_geo_cell": uf["geo_cell"],
            "user_age_bucket": uf["age_bucket"],
            "user_schedule_type": uf["schedule_type"],
            "user_interest_vector": uf["interest_vector"],
            "pos_event_id": pos_event_id,
            "pos_venue_id": pf["venue_id"],
            "pos_category": pf["category"],
            "pos_geo_cell": pf["geo_cell"],
            "pos_time_features": pf["time_features"],
            "pos_duration_norm": pf["duration_norm"],
            "pos_capacity_norm": pf["capacity_norm"],
            "pos_freshness_norm": pf["freshness_norm"],
        }


def collate_with_negatives(batch: list[dict], event_features: dict, n_neg: int = 10) -> dict:
    """Collate batch and sample negative events."""
    batch_size = len(batch)
    all_event_ids = list(event_features.keys())

    # Stack user features
    user_ids = torch.tensor([b["user_id"] for b in batch])
    user_geo_cells = torch.tensor([b["user_geo_cell"] for b in batch])
    user_age_buckets = torch.tensor([b["user_age_bucket"] for b in batch])
    user_schedule_types = torch.tensor([b["user_schedule_type"] for b in batch])
    user_interest_vectors = torch.stack([torch.tensor(b["user_interest_vector"]) for b in batch])

    # Stack positive event features
    pos_event_ids = torch.tensor([b["pos_event_id"] for b in batch])
    pos_venue_ids = torch.tensor([b["pos_venue_id"] for b in batch])
    pos_categories = torch.tensor([b["pos_category"] for b in batch])
    pos_geo_cells = torch.tensor([b["pos_geo_cell"] for b in batch])
    pos_time_features = torch.stack([torch.tensor(b["pos_time_features"]) for b in batch])
    pos_duration_norms = torch.tensor([b["pos_duration_norm"] for b in batch], dtype=torch.float32)
    pos_capacity_norms = torch.tensor([b["pos_capacity_norm"] for b in batch], dtype=torch.float32)
    pos_freshness_norms = torch.tensor([b["pos_freshness_norm"] for b in batch], dtype=torch.float32)

    # Sample negative events
    neg_event_ids = []
    neg_venue_ids = []
    neg_categories = []
    neg_geo_cells = []
    neg_time_features = []
    neg_duration_norms = []
    neg_capacity_norms = []
    neg_freshness_norms = []

    for i in range(batch_size):
        pos_id = batch[i]["pos_event_id"]
        negs = np.random.choice(
            [eid for eid in all_event_ids if eid != pos_id],
            size=n_neg,
            replace=False
        )
        for neg_id in negs:
            nf = event_features[neg_id]
            neg_event_ids.append(neg_id)
            neg_venue_ids.append(nf["venue_id"])
            neg_categories.append(nf["category"])
            neg_geo_cells.append(nf["geo_cell"])
            neg_time_features.append(nf["time_features"])
            neg_duration_norms.append(nf["duration_norm"])
            neg_capacity_norms.append(nf["capacity_norm"])
            neg_freshness_norms.append(nf["freshness_norm"])

    neg_event_ids = torch.tensor(neg_event_ids).reshape(batch_size, n_neg)
    neg_venue_ids = torch.tensor(neg_venue_ids).reshape(batch_size, n_neg)
    neg_categories = torch.tensor(neg_categories).reshape(batch_size, n_neg)
    neg_geo_cells = torch.tensor(neg_geo_cells).reshape(batch_size, n_neg)
    neg_time_features = torch.tensor(np.array(neg_time_features)).reshape(batch_size, n_neg, 4)
    neg_duration_norms = torch.tensor(neg_duration_norms, dtype=torch.float32).reshape(batch_size, n_neg)
    neg_capacity_norms = torch.tensor(neg_capacity_norms, dtype=torch.float32).reshape(batch_size, n_neg)
    neg_freshness_norms = torch.tensor(neg_freshness_norms, dtype=torch.float32).reshape(batch_size, n_neg)

    return {
        "user_features": {
            "user_id": user_ids,
            "geo_cell": user_geo_cells,
            "age_bucket": user_age_buckets,
            "schedule_type": user_schedule_types,
            "interest_vector": user_interest_vectors,
        },
        "pos_event_features": {
            "event_id": pos_event_ids,
            "venue_id": pos_venue_ids,
            "category": pos_categories,
            "geo_cell": pos_geo_cells,
            "time_features": pos_time_features,
            "duration_norm": pos_duration_norms,
            "capacity_norm": pos_capacity_norms,
            "freshness_norm": pos_freshness_norms,
        },
        "neg_event_features": {
            "event_id": neg_event_ids,
            "venue_id": neg_venue_ids,
            "category": neg_categories,
            "geo_cell": neg_geo_cells,
            "time_features": neg_time_features,
            "duration_norm": neg_duration_norms,
            "capacity_norm": neg_capacity_norms,
            "freshness_norm": neg_freshness_norms,
        },
    }


def train_epoch(
    model: TwoTowerModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move to device
        user_features = {k: v.to(device) for k, v in batch["user_features"].items()}
        pos_features = {k: v.to(device) for k, v in batch["pos_event_features"].items()}
        neg_features = {k: v.to(device) for k, v in batch["neg_event_features"].items()}

        # Get user embeddings
        user_emb = model.user_tower(
            user_ids=user_features["user_id"],
            geo_cells=user_features["geo_cell"],
            age_buckets=user_features["age_bucket"],
            schedule_types=user_features["schedule_type"],
            interest_vectors=user_features["interest_vector"],
        )

        # Get positive event embeddings
        pos_event_emb = model.event_tower(
            event_ids=pos_features["event_id"],
            venue_ids=pos_features["venue_id"],
            categories=pos_features["category"],
            geo_cells=pos_features["geo_cell"],
            time_features=pos_features["time_features"],
            duration_norm=pos_features["duration_norm"],
            capacity_norm=pos_features["capacity_norm"],
            freshness_norm=pos_features["freshness_norm"],
        )

        # Get negative event embeddings
        batch_size, n_neg = neg_features["event_id"].shape

        # Flatten negative features
        neg_event_emb = model.event_tower(
            event_ids=neg_features["event_id"].flatten(),
            venue_ids=neg_features["venue_id"].flatten(),
            categories=neg_features["category"].flatten(),
            geo_cells=neg_features["geo_cell"].flatten(),
            time_features=neg_features["time_features"].reshape(-1, 4),
            duration_norm=neg_features["duration_norm"].flatten(),
            capacity_norm=neg_features["capacity_norm"].flatten(),
            freshness_norm=neg_features["freshness_norm"].flatten(),
        )
        neg_event_emb = neg_event_emb.reshape(batch_size, n_neg, -1)

        # Compute BPR loss
        loss = model.compute_bpr_loss(user_emb, pos_event_emb, neg_event_emb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: TwoTowerModel,
    val_loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            user_features = {k: v.to(device) for k, v in batch["user_features"].items()}
            pos_features = {k: v.to(device) for k, v in batch["pos_event_features"].items()}
            neg_features = {k: v.to(device) for k, v in batch["neg_event_features"].items()}

            user_emb = model.user_tower(
                user_ids=user_features["user_id"],
                geo_cells=user_features["geo_cell"],
                age_buckets=user_features["age_bucket"],
                schedule_types=user_features["schedule_type"],
                interest_vectors=user_features["interest_vector"],
            )

            pos_event_emb = model.event_tower(
                event_ids=pos_features["event_id"],
                venue_ids=pos_features["venue_id"],
                categories=pos_features["category"],
                geo_cells=pos_features["geo_cell"],
                time_features=pos_features["time_features"],
                duration_norm=pos_features["duration_norm"],
                capacity_norm=pos_features["capacity_norm"],
                freshness_norm=pos_features["freshness_norm"],
            )

            batch_size, n_neg = neg_features["event_id"].shape
            neg_event_emb = model.event_tower(
                event_ids=neg_features["event_id"].flatten(),
                venue_ids=neg_features["venue_id"].flatten(),
                categories=neg_features["category"].flatten(),
                geo_cells=neg_features["geo_cell"].flatten(),
                time_features=neg_features["time_features"].reshape(-1, 4),
                duration_norm=neg_features["duration_norm"].flatten(),
                capacity_norm=neg_features["capacity_norm"].flatten(),
                freshness_norm=neg_features["freshness_norm"].flatten(),
            )
            neg_event_emb = neg_event_emb.reshape(batch_size, n_neg, -1)

            loss = model.compute_bpr_loss(user_emb, pos_event_emb, neg_event_emb)
            total_loss += loss.item()
            n_batches += 1

    return {"val_loss": total_loss / n_batches}


def main():
    parser = argparse.ArgumentParser(description="Train two-tower model")
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
    actions = pl.read_parquet(data_dir / "actions.parquet")

    # Filter actions by time split
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    train_end = base_time + timedelta(days=cfg.train.train_days[1])
    val_start = base_time + timedelta(days=cfg.train.val_days[0] - 1)
    val_end = base_time + timedelta(days=cfg.train.val_days[1])

    train_actions = actions.filter(pl.col("ts") < train_end)
    val_actions = actions.filter(
        (pl.col("ts") >= val_start) & (pl.col("ts") < val_end)
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TwoTowerDataset(train_actions, users, events, venues, cfg)
    val_dataset = TwoTowerDataset(val_actions, users, events, venues, cfg)

    # Create data loaders with custom collate
    collate_fn = lambda batch: collate_with_negatives(
        batch, train_dataset.event_features, cfg.train.tt_neg_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.tt_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_collate_fn = lambda batch: collate_with_negatives(
        batch, val_dataset.event_features, cfg.train.tt_neg_samples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.tt_batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=0,
    )

    # Create model
    logger.info("Creating model...")
    model = TwoTowerModel(
        n_users=len(users),
        n_events=len(events),
        n_venues=len(venues),
        n_categories=cfg.data.n_categories,
        n_geo_cells=cfg.data.n_geo_cells,
        interest_dim=cfg.data.n_interest_dims,
        embed_dim=cfg.model.user_embed_dim,
        hidden_dims=cfg.model.tower_hidden_dims,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.train.tt_lr,
        weight_decay=cfg.train.tt_weight_decay
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    metrics_history = []

    for epoch in range(cfg.train.tt_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        }
        metrics_history.append(metrics)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.train.tt_epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f}"
        )

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), models_dir / "two_tower_best.pt")

    # Save final model
    torch.save(model.state_dict(), models_dir / "two_tower_final.pt")

    # Extract and save embeddings
    logger.info("Extracting embeddings...")
    interest_cols = [f"interest_{i}" for i in range(cfg.data.n_interest_dims)]

    user_embeddings = extract_all_user_embeddings(model, users, interest_cols, device)
    event_embeddings = extract_all_event_embeddings(model, events, venues, device)

    torch.save(user_embeddings, models_dir / "user_embeddings.pt")
    torch.save(event_embeddings, models_dir / "event_embeddings.pt")

    logger.info(f"Saved user embeddings: {user_embeddings.shape}")
    logger.info(f"Saved event embeddings: {event_embeddings.shape}")

    # Save metrics
    with open(runs_dir / "two_tower_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
