"""
Synthetic data generator for calendar-aware social event feed.
Generates users, friendships, venues, events, calendars, impressions, and actions.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from loguru import logger
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config, DataConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def generate_users(cfg: DataConfig) -> pl.DataFrame:
    """Generate user data with interests and schedule preferences."""
    logger.info(f"Generating {cfg.n_users} users...")

    # Assign users to communities
    community_ids = np.random.randint(0, cfg.n_communities, size=cfg.n_users)

    # Generate interest vectors with community bias
    interest_vectors = np.random.randn(cfg.n_users, cfg.n_interest_dims) * 0.5

    # Add community-specific interest bias
    for c in range(cfg.n_communities):
        mask = community_ids == c
        # Each community has a random "preferred direction" in interest space
        community_bias = np.random.randn(cfg.n_interest_dims) * 1.5
        interest_vectors[mask] += community_bias

    # Normalize interest vectors
    norms = np.linalg.norm(interest_vectors, axis=1, keepdims=True)
    interest_vectors = interest_vectors / (norms + 1e-8)

    users = pl.DataFrame({
        "user_id": list(range(cfg.n_users)),
        "age_bucket": np.random.choice(
            ["18-24", "25-34", "35-44", "45-54", "55+"],
            size=cfg.n_users,
            p=[0.25, 0.35, 0.20, 0.12, 0.08]
        ),
        "home_geo_cell": np.random.randint(0, cfg.n_geo_cells, size=cfg.n_users),
        "schedule_type": np.random.choice(
            ["morning", "night", "flexible"],
            size=cfg.n_users,
            p=[0.3, 0.3, 0.4]
        ),
        "noise_level": np.random.uniform(0.1, 0.5, size=cfg.n_users),
        "community_id": community_ids,
    })

    # Add interest vector columns
    for i in range(cfg.n_interest_dims):
        users = users.with_columns(
            pl.lit(interest_vectors[:, i]).alias(f"interest_{i}")
        )

    logger.info(f"Generated users with {cfg.n_communities} communities")
    return users


def generate_friendships(cfg: DataConfig, users: pl.DataFrame) -> pl.DataFrame:
    """Generate friendship graph using stochastic block model."""
    logger.info("Generating friendship graph (stochastic block model)...")

    community_ids = users["community_id"].to_numpy()
    edges = []

    # Generate edges based on community membership
    for i in range(cfg.n_users):
        for j in range(i + 1, cfg.n_users):
            same_community = community_ids[i] == community_ids[j]
            prob = cfg.intra_community_prob if same_community else cfg.inter_community_prob

            if np.random.random() < prob:
                edges.append((i, j))

    logger.info(f"Generated {len(edges)} friendships")

    friendships = pl.DataFrame({
        "user_id_1": [e[0] for e in edges],
        "user_id_2": [e[1] for e in edges],
    })

    return friendships


def generate_venues(cfg: DataConfig) -> pl.DataFrame:
    """Generate venue data with categories and location."""
    logger.info(f"Generating {cfg.n_venues} venues...")

    # Generate multi-hot category vectors
    category_vectors = np.zeros((cfg.n_venues, cfg.n_categories), dtype=np.int32)
    for i in range(cfg.n_venues):
        n_cats = np.random.randint(1, 4)  # 1-3 categories per venue
        cats = np.random.choice(cfg.n_categories, size=n_cats, replace=False)
        category_vectors[i, cats] = 1

    venues = pl.DataFrame({
        "venue_id": list(range(cfg.n_venues)),
        "geo_cell": np.random.randint(0, cfg.n_geo_cells, size=cfg.n_venues),
        "popularity": np.random.pareto(2, size=cfg.n_venues) + 1,  # Power-law popularity
    })

    # Add category columns
    for i in range(cfg.n_categories):
        venues = venues.with_columns(
            pl.lit(category_vectors[:, i]).alias(f"category_{i}")
        )

    return venues


def generate_events(cfg: DataConfig, venues: pl.DataFrame) -> pl.DataFrame:
    """Generate events at venues over the horizon period."""
    logger.info(f"Generating {cfg.n_events} events over {cfg.horizon_days} days...")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    venue_ids = venues["venue_id"].to_numpy()
    venue_popularity = venues["popularity"].to_numpy()
    venue_probs = venue_popularity / venue_popularity.sum()

    # Assign events to venues based on popularity
    event_venue_ids = np.random.choice(venue_ids, size=cfg.n_events, p=venue_probs)

    # Generate start times (more events on weekends, evenings)
    start_times = []
    for _ in range(cfg.n_events):
        day_offset = np.random.randint(0, cfg.horizon_days)
        # Bias toward evening hours
        hour_probs = np.array([
            0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am
            0.03, 0.04, 0.05, 0.06, 0.08, 0.10,  # 6-11am
            0.12, 0.10, 0.08, 0.06, 0.05, 0.06,  # 12-5pm
            0.08, 0.10, 0.12, 0.10, 0.06, 0.04   # 6-11pm
        ])
        hour_probs = hour_probs / hour_probs.sum()
        hour = np.random.choice(24, p=hour_probs)
        minute = np.random.choice([0, 15, 30, 45])

        start_time = base_time + timedelta(days=int(day_offset), hours=int(hour), minutes=int(minute))
        start_times.append(start_time)

    # Generate other event attributes
    durations = np.random.choice([30, 60, 90, 120, 180, 240], size=cfg.n_events,
                                  p=[0.1, 0.3, 0.2, 0.25, 0.1, 0.05])
    categories = np.random.randint(0, cfg.n_categories, size=cfg.n_events)
    capacities = np.random.choice([20, 50, 100, 200, 500, 1000], size=cfg.n_events,
                                   p=[0.15, 0.3, 0.25, 0.15, 0.1, 0.05])

    # Freshness: when the event was posted (1-7 days before start)
    freshness_offsets = np.random.randint(1, 8, size=cfg.n_events)
    freshness_ts = [
        start_times[i] - timedelta(days=int(freshness_offsets[i]))
        for i in range(cfg.n_events)
    ]

    events = pl.DataFrame({
        "event_id": list(range(cfg.n_events)),
        "venue_id": event_venue_ids,
        "start_time": start_times,
        "duration_min": durations,
        "category": categories,
        "capacity": capacities,
        "freshness_ts": freshness_ts,
    })

    return events


def generate_calendars(cfg: DataConfig, users: pl.DataFrame) -> pl.DataFrame:
    """Generate busy blocks for each user's calendar."""
    logger.info(f"Generating calendars for {cfg.n_users} users over {cfg.horizon_days} days...")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    schedule_types = users["schedule_type"].to_list()
    user_ids = users["user_id"].to_list()

    calendar_entries = []

    for user_id, schedule_type in zip(user_ids, schedule_types):
        for day in range(cfg.horizon_days):
            # Number of busy blocks varies by schedule type
            if schedule_type == "morning":
                n_blocks = np.random.poisson(cfg.avg_busy_blocks_per_day * 1.2)
                # Morning people have more morning commitments
                hour_weights = np.array([0.05] * 6 + [0.15] * 6 + [0.08] * 6 + [0.04] * 6)
            elif schedule_type == "night":
                n_blocks = np.random.poisson(cfg.avg_busy_blocks_per_day * 0.8)
                # Night people have more afternoon/evening commitments
                hour_weights = np.array([0.02] * 6 + [0.05] * 6 + [0.15] * 6 + [0.12] * 6)
            else:
                n_blocks = np.random.poisson(cfg.avg_busy_blocks_per_day)
                hour_weights = np.array([0.03] * 6 + [0.10] * 6 + [0.10] * 6 + [0.08] * 6)

            hour_weights = hour_weights / hour_weights.sum()

            for _ in range(min(n_blocks, 8)):  # Cap at 8 blocks per day
                start_hour = np.random.choice(24, p=hour_weights)
                duration = np.random.randint(
                    cfg.busy_block_duration_min,
                    cfg.busy_block_duration_max + 1
                )

                block_start = base_time + timedelta(days=day, hours=start_hour)
                block_end = block_start + timedelta(minutes=duration)

                calendar_entries.append({
                    "user_id": user_id,
                    "block_start": block_start,
                    "block_end": block_end,
                    "day": day,
                })

    calendars = pl.DataFrame(calendar_entries)
    logger.info(f"Generated {len(calendar_entries)} calendar entries")
    return calendars


def compute_user_availability(
    user_id: int,
    calendars: pl.DataFrame,
    base_time: datetime,
    horizon_days: int
) -> np.ndarray:
    """Compute free time slots for a user as a bitset (15-min buckets)."""
    # 96 slots per day (15-min each)
    n_slots = horizon_days * 96
    free_slots = np.ones(n_slots, dtype=bool)

    user_blocks = calendars.filter(pl.col("user_id") == user_id)

    for row in user_blocks.iter_rows(named=True):
        block_start = row["block_start"]
        block_end = row["block_end"]

        # Convert to slot indices
        start_delta = (block_start - base_time).total_seconds() / 60 / 15
        end_delta = (block_end - base_time).total_seconds() / 60 / 15

        start_slot = max(0, int(start_delta))
        end_slot = min(n_slots, int(end_delta) + 1)

        free_slots[start_slot:end_slot] = False

    return free_slots


def check_availability(
    user_free_slots: np.ndarray,
    event_start: datetime,
    event_duration_min: int,
    base_time: datetime,
    travel_buffer_min: int = 30
) -> tuple[bool, float]:
    """Check if user is available for an event. Returns (is_available, next_feasible_hours)."""
    total_duration = event_duration_min + travel_buffer_min * 2

    # Convert event time to slot index
    event_delta = (event_start - base_time).total_seconds() / 60 / 15
    event_start_slot = int(event_delta)

    # Check slots needed
    n_slots_needed = int(np.ceil(total_duration / 15))
    event_end_slot = event_start_slot + n_slots_needed

    if event_start_slot < 0 or event_end_slot > len(user_free_slots):
        return False, -1.0

    # Check if all required slots are free
    is_available = np.all(user_free_slots[event_start_slot:event_end_slot])

    # Find next feasible slot if not available
    next_feasible_hours = -1.0
    if not is_available:
        # Look for next window of sufficient free time
        for i in range(event_end_slot, len(user_free_slots) - n_slots_needed):
            if np.all(user_free_slots[i:i + n_slots_needed]):
                next_feasible_hours = (i - event_start_slot) * 15 / 60
                break

    return is_available, next_feasible_hours


def compute_geo_distance(cell1: int, cell2: int, n_cells_per_side: int = 10) -> float:
    """Compute approximate distance between two geo cells."""
    x1, y1 = cell1 % n_cells_per_side, cell1 // n_cells_per_side
    x2, y2 = cell2 % n_cells_per_side, cell2 // n_cells_per_side
    # Each cell is ~2km
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 2.0


def generate_impressions_and_actions(
    cfg: DataConfig,
    users: pl.DataFrame,
    events: pl.DataFrame,
    venues: pl.DataFrame,
    calendars: pl.DataFrame,
    friendships: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate impressions and user actions based on probabilistic model."""
    logger.info("Generating impressions and actions...")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Precompute user interest vectors
    interest_cols = [f"interest_{i}" for i in range(cfg.n_interest_dims)]
    user_interests = users.select(interest_cols).to_numpy()

    # Precompute venue categories (convert to embeddings)
    category_cols = [f"category_{i}" for i in range(cfg.n_categories)]
    venue_categories = venues.select(category_cols).to_numpy()

    # Map event category to embedding (one-hot expanded)
    event_categories = events["category"].to_numpy()

    # Create category embeddings that match interest dimensions
    np.random.seed(cfg.seed + 1)
    category_to_interest = np.random.randn(cfg.n_categories, cfg.n_interest_dims)
    category_to_interest = category_to_interest / np.linalg.norm(
        category_to_interest, axis=1, keepdims=True
    )

    # Precompute user free slots
    logger.info("Precomputing user availability...")
    user_free_slots = {}
    for user_id in range(cfg.n_users):
        user_free_slots[user_id] = compute_user_availability(
            user_id, calendars, base_time, cfg.horizon_days
        )

    # Build friendship lookup
    friend_lookup = {i: set() for i in range(cfg.n_users)}
    for row in friendships.iter_rows(named=True):
        friend_lookup[row["user_id_1"]].add(row["user_id_2"])
        friend_lookup[row["user_id_2"]].add(row["user_id_1"])

    # Get venue geo cells
    venue_geo = dict(zip(venues["venue_id"].to_list(), venues["geo_cell"].to_list()))
    user_geo = dict(zip(users["user_id"].to_list(), users["home_geo_cell"].to_list()))

    impressions = []
    actions = []
    intent_lookup = {}  # Track who has intent for each event

    # Sample impressions (not all users see all events)
    logger.info("Sampling impressions...")
    event_data = events.to_dicts()

    for event_idx, event in enumerate(event_data):
        if event_idx % 1000 == 0:
            logger.info(f"Processing event {event_idx}/{len(event_data)}")

        event_id = event["event_id"]
        event_venue_id = event["venue_id"]
        event_start = event["start_time"]
        event_duration = event["duration_min"]
        event_cat = event["category"]
        event_geo = venue_geo[event_venue_id]
        event_cat_embedding = category_to_interest[event_cat]

        # Sample users who see this event (30% see each event on average)
        n_impressions = int(cfg.n_users * 0.3)
        impressed_users = np.random.choice(cfg.n_users, size=n_impressions, replace=False)

        event_intents = set()

        for user_id in impressed_users:
            user_id = int(user_id)

            # Compute match score
            interest_match = np.dot(user_interests[user_id], event_cat_embedding)

            # Compute distance penalty
            distance_km = compute_geo_distance(user_geo[user_id], event_geo)
            distance_penalty = np.exp(-cfg.distance_penalty_scale * distance_km)

            # Check availability
            is_available, next_feasible = check_availability(
                user_free_slots[user_id],
                event_start,
                event_duration,
                base_time
            )
            availability_mult = cfg.availability_boost if is_available else 0.3

            # Social proof: friends with intent
            friends = friend_lookup[user_id]
            friend_intents = len(event_intents.intersection(friends))
            social_mult = 1.0 + cfg.social_proof_boost * friend_intents

            # Base probability modifiers
            score = interest_match * distance_penalty * availability_mult * social_mult
            score = max(0.01, min(score, 5.0))  # Clamp

            # Generate impression timestamp (shortly after event posted)
            event_freshness = event["freshness_ts"]
            impression_offset = np.random.randint(0, 48)  # Within 48 hours of posting
            ts_shown = event_freshness + timedelta(hours=impression_offset)

            impressions.append({
                "user_id": user_id,
                "event_id": event_id,
                "ts_shown": ts_shown,
            })

            # Generate actions based on funnel
            user_noise = users["noise_level"][user_id]

            # View
            view_prob = cfg.base_view_prob * score * (1 + user_noise * np.random.randn() * 0.5)
            if np.random.random() < min(0.9, view_prob):
                ts_action = ts_shown + timedelta(minutes=np.random.randint(1, 60))
                actions.append({
                    "user_id": user_id,
                    "event_id": event_id,
                    "action_type": "view",
                    "ts": ts_action,
                })

                # Like (requires view)
                like_prob = cfg.base_like_prob * score
                if np.random.random() < min(0.5, like_prob):
                    ts_action = ts_action + timedelta(minutes=np.random.randint(1, 10))
                    actions.append({
                        "user_id": user_id,
                        "event_id": event_id,
                        "action_type": "like",
                        "ts": ts_action,
                    })

                    # Save (requires like usually)
                    if np.random.random() < min(0.3, cfg.base_save_prob * score):
                        ts_action = ts_action + timedelta(seconds=np.random.randint(10, 60))
                        actions.append({
                            "user_id": user_id,
                            "event_id": event_id,
                            "action_type": "save",
                            "ts": ts_action,
                        })

                    # Intent (strong signal, requires availability)
                    intent_prob = cfg.base_intent_prob * score * (2.0 if is_available else 0.1)
                    if np.random.random() < min(0.2, intent_prob):
                        ts_action = ts_action + timedelta(minutes=np.random.randint(1, 30))
                        actions.append({
                            "user_id": user_id,
                            "event_id": event_id,
                            "action_type": "intent",
                            "ts": ts_action,
                        })
                        event_intents.add(user_id)

                        # Attend (requires intent)
                        if is_available and np.random.random() < min(0.5, cfg.base_attend_prob * score * 3):
                            ts_action = event_start + timedelta(minutes=np.random.randint(0, event_duration))
                            actions.append({
                                "user_id": user_id,
                                "event_id": event_id,
                                "action_type": "attend",
                                "ts": ts_action,
                            })

        intent_lookup[event_id] = event_intents

    logger.info(f"Generated {len(impressions)} impressions and {len(actions)} actions")

    impressions_df = pl.DataFrame(impressions)
    actions_df = pl.DataFrame(actions)

    return impressions_df, actions_df


def save_data(
    cfg: DataConfig,
    users: pl.DataFrame,
    friendships: pl.DataFrame,
    venues: pl.DataFrame,
    events: pl.DataFrame,
    calendars: pl.DataFrame,
    impressions: pl.DataFrame,
    actions: pl.DataFrame
) -> None:
    """Save all generated data to parquet files."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    users.write_parquet(cfg.data_dir / "users.parquet")
    friendships.write_parquet(cfg.data_dir / "friendships.parquet")
    venues.write_parquet(cfg.data_dir / "venues.parquet")
    events.write_parquet(cfg.data_dir / "events.parquet")
    calendars.write_parquet(cfg.data_dir / "calendars.parquet")
    impressions.write_parquet(cfg.data_dir / "impressions.parquet")
    actions.write_parquet(cfg.data_dir / "actions.parquet")

    logger.info(f"Saved all data to {cfg.data_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--small", action="store_true", help="Use small config for testing")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    cfg = get_config(small=args.small)

    if args.seed is not None:
        cfg.data.seed = args.seed

    set_seed(cfg.data.seed)

    logger.info(f"Starting data generation (small={args.small})")
    logger.info(f"Config: {cfg.data.n_users} users, {cfg.data.n_events} events")

    # Generate all data
    users = generate_users(cfg.data)
    friendships = generate_friendships(cfg.data, users)
    venues = generate_venues(cfg.data)
    events = generate_events(cfg.data, venues)
    calendars = generate_calendars(cfg.data, users)
    impressions, actions = generate_impressions_and_actions(
        cfg.data, users, events, venues, calendars, friendships
    )

    # Save data
    save_data(cfg.data, users, friendships, venues, events, calendars, impressions, actions)

    # Print summary statistics
    logger.info("\n=== Data Generation Summary ===")
    logger.info(f"Users: {len(users)}")
    logger.info(f"Friendships: {len(friendships)}")
    logger.info(f"Venues: {len(venues)}")
    logger.info(f"Events: {len(events)}")
    logger.info(f"Calendar entries: {len(calendars)}")
    logger.info(f"Impressions: {len(impressions)}")
    logger.info(f"Actions: {len(actions)}")

    # Action breakdown
    if len(actions) > 0:
        action_counts = actions.group_by("action_type").agg(pl.len().alias("count"))
        logger.info(f"\nAction breakdown:\n{action_counts}")


if __name__ == "__main__":
    main()
