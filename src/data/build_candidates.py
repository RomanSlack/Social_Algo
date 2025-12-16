"""
Build candidate (user, event) pairs with engineered features for training.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config


def compute_geo_distance(cell1: int, cell2: int, n_cells_per_side: int = 10) -> float:
    """Compute approximate distance between two geo cells in km."""
    x1, y1 = cell1 % n_cells_per_side, cell1 // n_cells_per_side
    x2, y2 = cell2 % n_cells_per_side, cell2 // n_cells_per_side
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 2.0


def build_friend_lookup(friendships: pl.DataFrame) -> dict[int, set[int]]:
    """Build friend adjacency lookup."""
    friend_lookup = {}
    for row in friendships.iter_rows(named=True):
        u1, u2 = row["user_id_1"], row["user_id_2"]
        if u1 not in friend_lookup:
            friend_lookup[u1] = set()
        if u2 not in friend_lookup:
            friend_lookup[u2] = set()
        friend_lookup[u1].add(u2)
        friend_lookup[u2].add(u1)
    return friend_lookup


def build_intent_lookup(actions: pl.DataFrame) -> dict[int, set[int]]:
    """Build lookup of users who have intent for each event."""
    intent_actions = actions.filter(pl.col("action_type") == "intent")
    intent_lookup = {}
    for row in intent_actions.iter_rows(named=True):
        event_id = row["event_id"]
        user_id = row["user_id"]
        if event_id not in intent_lookup:
            intent_lookup[event_id] = set()
        intent_lookup[event_id].add(user_id)
    return intent_lookup


def check_availability_fit(
    user_free_indices: set[int],
    event_start_bucket: int,
    n_buckets_needed: int,
    n_total_buckets: int
) -> tuple[bool, float]:
    """Check if user is available for event."""
    if event_start_bucket < 0 or event_start_bucket + n_buckets_needed > n_total_buckets:
        return False, -1.0

    required_buckets = set(range(event_start_bucket, event_start_bucket + n_buckets_needed))
    is_available = required_buckets.issubset(user_free_indices)

    # Find next feasible slot
    next_feasible_hours = -1.0
    if not is_available:
        for start in range(event_start_bucket + n_buckets_needed, n_total_buckets - n_buckets_needed):
            candidate_buckets = set(range(start, start + n_buckets_needed))
            if candidate_buckets.issubset(user_free_indices):
                next_feasible_hours = (start - event_start_bucket) * 15 / 60
                break

    return is_available, next_feasible_hours


def build_candidate_features(
    users: pl.DataFrame,
    events: pl.DataFrame,
    venues: pl.DataFrame,
    actions: pl.DataFrame,
    friendships: pl.DataFrame,
    free_buckets_df: pl.DataFrame,
    impressions: pl.DataFrame,
    cfg,
    base_time: datetime,
    include_labels: bool = True,
    day_filter: tuple[int, int] | None = None
) -> pl.DataFrame:
    """
    Build features for (user, event) candidate pairs.

    Args:
        day_filter: Optional (start_day, end_day) to filter impressions by day.
    """
    logger.info("Building candidate features...")

    # Build lookups
    friend_lookup = build_friend_lookup(friendships)
    intent_lookup = build_intent_lookup(actions)

    # User geo mapping
    user_geo = dict(zip(users["user_id"].to_list(), users["home_geo_cell"].to_list()))

    # Venue geo mapping
    venue_geo = dict(zip(venues["venue_id"].to_list(), venues["geo_cell"].to_list()))

    # Event -> venue mapping
    event_venue = dict(zip(events["event_id"].to_list(), events["venue_id"].to_list()))
    event_start = dict(zip(events["event_id"].to_list(), events["start_time"].to_list()))
    event_duration = dict(zip(events["event_id"].to_list(), events["duration_min"].to_list()))
    event_freshness = dict(zip(events["event_id"].to_list(), events["freshness_ts"].to_list()))
    event_category = dict(zip(events["event_id"].to_list(), events["category"].to_list()))

    # User interest vectors
    interest_cols = [f"interest_{i}" for i in range(cfg.data.n_interest_dims)]
    user_interests = {}
    for row in users.iter_rows(named=True):
        user_interests[row["user_id"]] = np.array([row[c] for c in interest_cols])

    # Category embeddings (same as in gen_synth)
    np.random.seed(cfg.data.seed + 1)
    category_to_interest = np.random.randn(cfg.data.n_categories, cfg.data.n_interest_dims)
    category_to_interest = category_to_interest / np.linalg.norm(
        category_to_interest, axis=1, keepdims=True
    )

    # Free buckets
    n_buckets_per_day = 24 * 4  # 15-min buckets
    n_total_buckets = cfg.data.horizon_days * n_buckets_per_day
    travel_buffer_buckets = 2  # 30 min buffer

    user_free_indices = {}
    for row in free_buckets_df.iter_rows(named=True):
        user_free_indices[row["user_id"]] = set(row["free_bucket_indices"])

    # Action labels
    if include_labels:
        action_labels = {}
        for row in actions.iter_rows(named=True):
            key = (row["user_id"], row["event_id"])
            action_type = row["action_type"]
            if key not in action_labels:
                action_labels[key] = {"view": 0, "like": 0, "save": 0, "intent": 0, "attend": 0}
            action_labels[key][action_type] = 1

    # Filter impressions by day if specified
    if day_filter is not None:
        start_day, end_day = day_filter
        start_time = base_time + timedelta(days=start_day - 1)
        end_time = base_time + timedelta(days=end_day)
        impressions = impressions.filter(
            (pl.col("ts_shown") >= start_time) & (pl.col("ts_shown") < end_time)
        )

    # Build features for each impression
    candidates = []
    for row in impressions.iter_rows(named=True):
        user_id = row["user_id"]
        event_id = row["event_id"]
        ts_shown = row["ts_shown"]

        # Skip if missing data
        if user_id not in user_geo or event_id not in event_venue:
            continue

        venue_id = event_venue[event_id]
        if venue_id not in venue_geo:
            continue

        # 1. Distance
        user_cell = user_geo[user_id]
        event_cell = venue_geo[venue_id]
        distance_km = compute_geo_distance(user_cell, event_cell)

        # 2. Availability fit
        evt_start = event_start[event_id]
        evt_duration = event_duration[event_id]
        total_duration_buckets = int(np.ceil((evt_duration + 60) / 15))  # +60 for travel

        start_delta_min = (evt_start - base_time).total_seconds() / 60 - 30
        start_bucket = int(start_delta_min // 15)

        user_free = user_free_indices.get(user_id, set())
        is_available, next_feasible = check_availability_fit(
            user_free, start_bucket, total_duration_buckets, n_total_buckets
        )

        # 3. Freshness
        evt_fresh = event_freshness[event_id]
        freshness_hours = (ts_shown - evt_fresh).total_seconds() / 3600

        # 4. Category match
        user_vec = user_interests.get(user_id)
        evt_cat = event_category[event_id]
        if user_vec is not None and evt_cat < len(category_to_interest):
            cat_vec = category_to_interest[evt_cat]
            category_match = float(np.dot(user_vec, cat_vec))
        else:
            category_match = 0.0

        # 5. Friend features
        friends = friend_lookup.get(user_id, set())
        event_intents = intent_lookup.get(event_id, set())

        friend_intent_count = len(friends.intersection(event_intents))

        # Friend feasible count (approximate - check a sample)
        friend_feasible_count = 0
        for friend_id in friends:
            friend_free = user_free_indices.get(friend_id, set())
            friend_avail, _ = check_availability_fit(
                friend_free, start_bucket, total_duration_buckets, n_total_buckets
            )
            if friend_avail:
                friend_feasible_count += 1

        candidate = {
            "user_id": user_id,
            "event_id": event_id,
            "ts_shown": ts_shown,
            "availability_fit": 1 if is_available else 0,
            "next_feasible_hours": next_feasible,
            "distance_km": distance_km,
            "friend_intent_count": friend_intent_count,
            "friend_feasible_count": friend_feasible_count,
            "freshness_hours": freshness_hours,
            "category_match": category_match,
        }

        # Add labels if available
        if include_labels:
            key = (user_id, event_id)
            if key in action_labels:
                candidate.update(action_labels[key])
            else:
                candidate.update({"view": 0, "like": 0, "save": 0, "intent": 0, "attend": 0})

        candidates.append(candidate)

    logger.info(f"Built {len(candidates)} candidate features")
    return pl.DataFrame(candidates)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build candidate features")
    parser.add_argument("--small", action="store_true", help="Use small config")
    args = parser.parse_args()

    cfg = get_config(small=args.small)
    data_dir = cfg.data.data_dir

    # Load data
    logger.info("Loading data...")
    users = pl.read_parquet(data_dir / "users.parquet")
    events = pl.read_parquet(data_dir / "events.parquet")
    venues = pl.read_parquet(data_dir / "venues.parquet")
    actions = pl.read_parquet(data_dir / "actions.parquet")
    friendships = pl.read_parquet(data_dir / "friendships.parquet")
    free_buckets_df = pl.read_parquet(data_dir / "free_buckets.parquet")
    impressions = pl.read_parquet(data_dir / "impressions.parquet")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Build train/val/test splits by time
    train_start, train_end = cfg.train.train_days
    val_start, val_end = cfg.train.val_days
    test_start, test_end = cfg.train.test_days

    # Train set
    logger.info(f"Building train set (days {train_start}-{train_end})...")
    train_candidates = build_candidate_features(
        users, events, venues, actions, friendships, free_buckets_df,
        impressions, cfg, base_time, include_labels=True,
        day_filter=(train_start, train_end)
    )
    train_candidates.write_parquet(data_dir / "train_candidates.parquet")
    logger.info(f"Saved {len(train_candidates)} train candidates")

    # Val set
    logger.info(f"Building val set (days {val_start}-{val_end})...")
    val_candidates = build_candidate_features(
        users, events, venues, actions, friendships, free_buckets_df,
        impressions, cfg, base_time, include_labels=True,
        day_filter=(val_start, val_end)
    )
    val_candidates.write_parquet(data_dir / "val_candidates.parquet")
    logger.info(f"Saved {len(val_candidates)} val candidates")

    # Test set
    logger.info(f"Building test set (days {test_start}-{test_end})...")
    test_candidates = build_candidate_features(
        users, events, venues, actions, friendships, free_buckets_df,
        impressions, cfg, base_time, include_labels=True,
        day_filter=(test_start, test_end)
    )
    test_candidates.write_parquet(data_dir / "test_candidates.parquet")
    logger.info(f"Saved {len(test_candidates)} test candidates")

    # Print statistics
    logger.info("\n=== Candidate Feature Statistics ===")
    logger.info(f"Train samples: {len(train_candidates)}")
    logger.info(f"Val samples: {len(val_candidates)}")
    logger.info(f"Test samples: {len(test_candidates)}")

    if len(train_candidates) > 0:
        intent_rate = train_candidates["intent"].mean()
        avail_rate = train_candidates["availability_fit"].mean()
        logger.info(f"Train intent rate: {intent_rate:.4f}")
        logger.info(f"Train availability rate: {avail_rate:.4f}")


if __name__ == "__main__":
    main()
