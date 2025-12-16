"""
Derive user availability buckets and event feasibility windows.
Computes free-time analysis for users and matching with event times.
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


def compute_free_buckets(
    calendars: pl.DataFrame,
    n_users: int,
    base_time: datetime,
    horizon_days: int,
    bucket_size_min: int = 15
) -> dict[int, np.ndarray]:
    """
    Compute free time buckets for all users.

    Returns dict mapping user_id -> boolean array of free buckets.
    Each bucket represents bucket_size_min minutes.
    """
    logger.info(f"Computing free buckets for {n_users} users...")

    n_buckets_per_day = 24 * 60 // bucket_size_min
    n_total_buckets = horizon_days * n_buckets_per_day

    # Initialize all users as free
    user_free_buckets = {
        user_id: np.ones(n_total_buckets, dtype=bool)
        for user_id in range(n_users)
    }

    # Mark busy periods
    for row in calendars.iter_rows(named=True):
        user_id = row["user_id"]
        block_start = row["block_start"]
        block_end = row["block_end"]

        # Convert to bucket indices
        start_delta_min = (block_start - base_time).total_seconds() / 60
        end_delta_min = (block_end - base_time).total_seconds() / 60

        start_bucket = max(0, int(start_delta_min // bucket_size_min))
        end_bucket = min(n_total_buckets, int(np.ceil(end_delta_min / bucket_size_min)))

        if user_id in user_free_buckets:
            user_free_buckets[user_id][start_bucket:end_bucket] = False

    # Compute statistics
    free_fractions = [
        np.mean(buckets) for buckets in user_free_buckets.values()
    ]
    logger.info(f"Average free fraction: {np.mean(free_fractions):.2%}")
    logger.info(f"Min free fraction: {np.min(free_fractions):.2%}")
    logger.info(f"Max free fraction: {np.max(free_fractions):.2%}")

    return user_free_buckets


def compute_event_feasibility(
    events: pl.DataFrame,
    user_free_buckets: dict[int, np.ndarray],
    base_time: datetime,
    bucket_size_min: int = 15,
    travel_buffer_min: int = 30
) -> pl.DataFrame:
    """
    For each (user, event) pair, compute whether the event is feasible.

    Returns DataFrame with feasibility metrics.
    """
    logger.info("Computing event feasibility for all users...")

    n_users = len(user_free_buckets)
    n_total_buckets = len(list(user_free_buckets.values())[0])

    results = []

    for event in events.iter_rows(named=True):
        event_id = event["event_id"]
        event_start = event["start_time"]
        event_duration = event["duration_min"]

        # Total time needed including travel buffer
        total_duration = event_duration + 2 * travel_buffer_min
        n_buckets_needed = int(np.ceil(total_duration / bucket_size_min))

        # Convert event time to bucket index (including pre-travel buffer)
        start_delta_min = (event_start - base_time).total_seconds() / 60 - travel_buffer_min
        start_bucket = int(start_delta_min // bucket_size_min)
        end_bucket = start_bucket + n_buckets_needed

        # For efficiency, sample users instead of checking all
        sampled_users = list(user_free_buckets.keys())

        feasible_count = 0
        for user_id in sampled_users:
            buckets = user_free_buckets[user_id]

            if start_bucket < 0 or end_bucket > n_total_buckets:
                continue

            if np.all(buckets[start_bucket:end_bucket]):
                feasible_count += 1

        results.append({
            "event_id": event_id,
            "feasible_user_count": feasible_count,
            "feasibility_rate": feasible_count / n_users if n_users > 0 else 0,
        })

    feasibility_df = pl.DataFrame(results)

    # Log statistics
    avg_rate = feasibility_df["feasibility_rate"].mean()
    logger.info(f"Average event feasibility rate: {avg_rate:.2%}")

    return feasibility_df


def compute_schedule_embedding(
    free_buckets: np.ndarray,
    horizon_days: int,
    bucket_size_min: int = 15
) -> np.ndarray:
    """
    Compute a schedule embedding from free buckets.

    Returns a 7x24 matrix (day-of-week x hour-of-day) aggregated free probability.
    Flattened to 168 dimensions.
    """
    buckets_per_hour = 60 // bucket_size_min
    buckets_per_day = 24 * buckets_per_hour

    # Aggregate by hour
    schedule_matrix = np.zeros((7, 24))
    counts = np.zeros((7, 24))

    for day in range(horizon_days):
        dow = day % 7  # Day of week
        day_start = day * buckets_per_day

        for hour in range(24):
            hour_start = day_start + hour * buckets_per_hour
            hour_end = hour_start + buckets_per_hour

            if hour_end <= len(free_buckets):
                schedule_matrix[dow, hour] += np.mean(free_buckets[hour_start:hour_end])
                counts[dow, hour] += 1

    # Average over days
    with np.errstate(divide='ignore', invalid='ignore'):
        schedule_matrix = np.divide(schedule_matrix, counts, where=counts > 0)
        schedule_matrix = np.nan_to_num(schedule_matrix, nan=0.5)

    return schedule_matrix.flatten()


def compute_all_schedule_embeddings(
    user_free_buckets: dict[int, np.ndarray],
    horizon_days: int
) -> pl.DataFrame:
    """Compute schedule embeddings for all users."""
    logger.info("Computing schedule embeddings...")

    embeddings = []
    for user_id, buckets in user_free_buckets.items():
        emb = compute_schedule_embedding(buckets, horizon_days)
        embeddings.append({"user_id": user_id, **{f"sched_{i}": v for i, v in enumerate(emb)}})

    return pl.DataFrame(embeddings)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Derive availability features")
    parser.add_argument("--small", action="store_true", help="Use small config")
    args = parser.parse_args()

    cfg = get_config(small=args.small)
    data_dir = cfg.data.data_dir

    # Load data
    logger.info("Loading data...")
    calendars = pl.read_parquet(data_dir / "calendars.parquet")
    events = pl.read_parquet(data_dir / "events.parquet")
    users = pl.read_parquet(data_dir / "users.parquet")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    n_users = len(users)

    # Compute free buckets
    user_free_buckets = compute_free_buckets(
        calendars, n_users, base_time, cfg.data.horizon_days
    )

    # Compute schedule embeddings
    schedule_embs = compute_all_schedule_embeddings(user_free_buckets, cfg.data.horizon_days)
    schedule_embs.write_parquet(data_dir / "schedule_embeddings.parquet")
    logger.info(f"Saved schedule embeddings to {data_dir / 'schedule_embeddings.parquet'}")

    # Compute event feasibility
    feasibility = compute_event_feasibility(events, user_free_buckets, base_time)
    feasibility.write_parquet(data_dir / "event_feasibility.parquet")
    logger.info(f"Saved event feasibility to {data_dir / 'event_feasibility.parquet'}")

    # Save free buckets as sparse representation (for serving)
    logger.info("Saving free bucket indices...")
    free_bucket_data = []
    for user_id, buckets in user_free_buckets.items():
        free_indices = np.where(buckets)[0].tolist()
        free_bucket_data.append({
            "user_id": user_id,
            "free_bucket_indices": free_indices,
        })

    free_buckets_df = pl.DataFrame(free_bucket_data)
    free_buckets_df.write_parquet(data_dir / "free_buckets.parquet")
    logger.info(f"Saved free buckets to {data_dir / 'free_buckets.parquet'}")


if __name__ == "__main__":
    main()
