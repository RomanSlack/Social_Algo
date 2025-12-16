"""
Flask web application for browsing the event feed recommender.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import polars as pl
import torch
from flask import Flask, render_template, request, jsonify
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.models.two_tower import TwoTowerModel
from src.models.reranker import RerankerMLP
from src.models.gnn import TemporalHeteroGNN


# Global state
app = Flask(__name__, template_folder="templates")

# Will be populated on startup
state = {
    "users": None,
    "events": None,
    "venues": None,
    "friendships": None,
    "actions": None,
    "free_buckets": None,
    "user_embeddings": None,
    "event_embeddings": None,
    "gnn_user_embeddings": None,
    "gnn_event_embeddings": None,
    "reranker": None,
    "friend_lookup": None,
    "intent_lookup": None,
    "cfg": None,
}


def load_data(cfg):
    """Load all required data."""
    data_dir = cfg.data.data_dir
    models_dir = cfg.train.models_dir

    state["users"] = pl.read_parquet(data_dir / "users.parquet")
    state["events"] = pl.read_parquet(data_dir / "events.parquet")
    state["venues"] = pl.read_parquet(data_dir / "venues.parquet")
    state["friendships"] = pl.read_parquet(data_dir / "friendships.parquet")
    state["actions"] = pl.read_parquet(data_dir / "actions.parquet")
    state["free_buckets"] = pl.read_parquet(data_dir / "free_buckets.parquet")

    # Load embeddings
    state["user_embeddings"] = torch.load(models_dir / "user_embeddings.pt")
    state["event_embeddings"] = torch.load(models_dir / "event_embeddings.pt")

    if (models_dir / "gnn_user_embeddings.pt").exists():
        state["gnn_user_embeddings"] = torch.load(models_dir / "gnn_user_embeddings.pt")
        state["gnn_event_embeddings"] = torch.load(models_dir / "gnn_event_embeddings.pt")

    # Load reranker
    if (models_dir / "reranker_best.pt").exists():
        reranker = RerankerMLP(
            n_engineered_features=7,
            hidden_dims=cfg.model.reranker_hidden_dims,
            use_gnn_embeddings=state["gnn_user_embeddings"] is not None,
            gnn_embed_dim=cfg.model.gnn_hidden_dim,
        )
        reranker.load_state_dict(torch.load(models_dir / "reranker_best.pt", map_location="cpu"))
        reranker.eval()
        state["reranker"] = reranker

    # Build lookups
    state["friend_lookup"] = build_friend_lookup(state["friendships"])
    state["intent_lookup"] = build_intent_lookup(state["actions"])

    state["cfg"] = cfg

    logger.info(f"Loaded {len(state['users'])} users and {len(state['events'])} events")


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


def compute_geo_distance(cell1: int, cell2: int, n_cells_per_side: int = 10) -> float:
    """Compute approximate distance between two geo cells in km."""
    x1, y1 = cell1 % n_cells_per_side, cell1 // n_cells_per_side
    x2, y2 = cell2 % n_cells_per_side, cell2 // n_cells_per_side
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * 2.0


def check_availability(user_id: int, event_id: int) -> tuple[bool, float]:
    """Check if user is available for event."""
    cfg = state["cfg"]
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Get user free buckets
    user_row = state["free_buckets"].filter(pl.col("user_id") == user_id)
    if len(user_row) == 0:
        return False, -1.0

    free_indices = set(user_row["free_bucket_indices"][0])

    # Get event time
    event_row = state["events"].filter(pl.col("event_id") == event_id)
    if len(event_row) == 0:
        return False, -1.0

    event_start = event_row["start_time"][0]
    event_duration = event_row["duration_min"][0]

    # Convert to buckets
    total_duration = event_duration + 60  # Travel buffer
    n_buckets_needed = int(np.ceil(total_duration / 15))

    start_delta_min = (event_start - base_time).total_seconds() / 60 - 30
    start_bucket = int(start_delta_min // 15)

    n_total_buckets = cfg.data.horizon_days * 96  # 15-min buckets

    if start_bucket < 0 or start_bucket + n_buckets_needed > n_total_buckets:
        return False, -1.0

    required_buckets = set(range(start_bucket, start_bucket + n_buckets_needed))
    is_available = required_buckets.issubset(free_indices)

    # Find next feasible slot
    next_feasible_hours = -1.0
    if not is_available:
        for start in range(start_bucket + n_buckets_needed, n_total_buckets - n_buckets_needed):
            candidate_buckets = set(range(start, start + n_buckets_needed))
            if candidate_buckets.issubset(free_indices):
                next_feasible_hours = (start - start_bucket) * 15 / 60
                break

    return is_available, next_feasible_hours


def get_user_feed(user_id: int, top_k: int = 50) -> list[dict]:
    """Generate personalized feed for a user."""
    cfg = state["cfg"]

    # Get user info
    user_row = state["users"].filter(pl.col("user_id") == user_id)
    if len(user_row) == 0:
        return []

    user_geo = user_row["home_geo_cell"][0]

    # Get user embeddings
    user_emb = state["user_embeddings"][user_id].numpy()
    event_embs = state["event_embeddings"].numpy()

    # Compute two-tower similarities
    tt_scores = np.dot(event_embs, user_emb)

    # Get venue geo mapping
    venue_geo = dict(zip(state["venues"]["venue_id"].to_list(), state["venues"]["geo_cell"].to_list()))

    # Build candidate features
    candidates = []
    friends = state["friend_lookup"].get(user_id, set())

    for event_id in range(len(state["events"])):
        event_row = state["events"].filter(pl.col("event_id") == event_id)
        if len(event_row) == 0:
            continue

        venue_id = event_row["venue_id"][0]
        event_geo = venue_geo.get(venue_id, 0)

        # Features
        is_available, next_feasible = check_availability(user_id, event_id)
        distance_km = compute_geo_distance(user_geo, event_geo)

        event_intents = state["intent_lookup"].get(event_id, set())
        friend_intent_count = len(friends.intersection(event_intents))

        # Friend feasible count
        friend_feasible_count = 0
        for friend_id in list(friends)[:20]:  # Limit for speed
            friend_avail, _ = check_availability(friend_id, event_id)
            if friend_avail:
                friend_feasible_count += 1

        candidates.append({
            "event_id": event_id,
            "tt_score": float(tt_scores[event_id]),
            "availability_fit": 1 if is_available else 0,
            "next_feasible_hours": next_feasible,
            "distance_km": distance_km,
            "friend_intent_count": friend_intent_count,
            "friend_feasible_count": friend_feasible_count,
        })

    # Apply reranker if available
    if state["reranker"] is not None and len(candidates) > 0:
        # Prepare features
        tt_sims = torch.tensor([c["tt_score"] for c in candidates], dtype=torch.float32)
        features = torch.tensor([
            [
                c["availability_fit"],
                max(0, c["next_feasible_hours"]) / 24 if c["next_feasible_hours"] > 0 else 0,
                c["distance_km"] / 20,
                c["friend_intent_count"] / 5,
                c["friend_feasible_count"] / 10,
                0,  # freshness placeholder
                0,  # category match placeholder
            ]
            for c in candidates
        ], dtype=torch.float32)

        # GNN embeddings if available
        gnn_user_emb = None
        gnn_event_emb = None
        if state["gnn_user_embeddings"] is not None:
            gnn_user_emb = state["gnn_user_embeddings"][user_id].unsqueeze(0).expand(len(candidates), -1)
            event_ids = [c["event_id"] for c in candidates]
            gnn_event_emb = state["gnn_event_embeddings"][event_ids]

        with torch.no_grad():
            scores = state["reranker"](tt_sims, features, gnn_user_emb, gnn_event_emb)
            scores = scores.numpy()

        for i, c in enumerate(candidates):
            c["final_score"] = float(scores[i])
    else:
        # Fallback: simple weighted score
        for c in candidates:
            c["final_score"] = (
                c["tt_score"] * 0.3 +
                c["availability_fit"] * 0.4 +
                c["friend_intent_count"] * 0.2 +
                max(0, 1 - c["distance_km"] / 20) * 0.1
            )

    # Sort by final score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # Normalize scores to 0-100 range for better UX
    if len(candidates) > 1:
        scores = [c["final_score"] for c in candidates]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1
        for c in candidates:
            c["display_score"] = round(100 * (c["final_score"] - min_score) / score_range, 1)
    else:
        for c in candidates:
            c["display_score"] = 50.0

    # Add event details
    result = []
    for c in candidates[:top_k]:
        event_row = state["events"].filter(pl.col("event_id") == c["event_id"])
        if len(event_row) == 0:
            continue

        venue_id = event_row["venue_id"][0]
        venue_row = state["venues"].filter(pl.col("venue_id") == venue_id)

        result.append({
            "event_id": c["event_id"],
            "venue_id": venue_id,
            "start_time": str(event_row["start_time"][0]),
            "duration_min": int(event_row["duration_min"][0]),
            "category": int(event_row["category"][0]),
            "capacity": int(event_row["capacity"][0]),
            "final_score": round(c["final_score"], 4),
            "display_score": c["display_score"],
            "tt_score": round(c["tt_score"], 4),
            "availability_fit": c["availability_fit"],
            "next_feasible_hours": round(c["next_feasible_hours"], 1) if c["next_feasible_hours"] > 0 else None,
            "distance_km": round(c["distance_km"], 1),
            "friend_intent_count": c["friend_intent_count"],
            "friend_feasible_count": c["friend_feasible_count"],
            "why": get_top_factors(c),
        })

    return result


def get_top_factors(candidate: dict) -> list[dict]:
    """Get top contributing factors for ranking."""
    factors = [
        {"name": "Availability", "value": candidate["availability_fit"], "impact": "high" if candidate["availability_fit"] else "low"},
        {"name": "Friend Intent", "value": candidate["friend_intent_count"], "impact": "high" if candidate["friend_intent_count"] > 0 else "neutral"},
        {"name": "Friend Feasible", "value": candidate["friend_feasible_count"], "impact": "high" if candidate["friend_feasible_count"] > 2 else "neutral"},
        {"name": "Distance (km)", "value": round(candidate["distance_km"], 1), "impact": "high" if candidate["distance_km"] < 5 else "low"},
        {"name": "Content Match", "value": round(candidate["tt_score"], 3), "impact": "high" if candidate["tt_score"] > 0.5 else "neutral"},
    ]
    # Sort by impact
    impact_order = {"high": 0, "neutral": 1, "low": 2}
    factors.sort(key=lambda x: impact_order[x["impact"]])
    return factors[:5]


def get_event_details(event_id: int, user_id: int) -> dict:
    """Get detailed event info for a user."""
    event_row = state["events"].filter(pl.col("event_id") == event_id)
    if len(event_row) == 0:
        return {}

    venue_id = event_row["venue_id"][0]
    venue_row = state["venues"].filter(pl.col("venue_id") == venue_id)

    is_available, next_feasible = check_availability(user_id, event_id)

    # Get friends who can attend
    friends = state["friend_lookup"].get(user_id, set())
    event_intents = state["intent_lookup"].get(event_id, set())

    friends_info = []
    for friend_id in friends:
        friend_avail, friend_next = check_availability(friend_id, event_id)
        friend_has_intent = friend_id in event_intents

        friends_info.append({
            "user_id": friend_id,
            "is_available": friend_avail,
            "has_intent": friend_has_intent,
            "next_feasible_hours": round(friend_next, 1) if friend_next > 0 else None,
        })

    # Sort: intent + available first, then available, then intent
    friends_info.sort(key=lambda x: (
        -(x["has_intent"] and x["is_available"]),
        -x["is_available"],
        -x["has_intent"]
    ))

    return {
        "event_id": event_id,
        "venue_id": venue_id,
        "venue_geo": venue_row["geo_cell"][0] if len(venue_row) > 0 else None,
        "start_time": str(event_row["start_time"][0]),
        "duration_min": int(event_row["duration_min"][0]),
        "category": int(event_row["category"][0]),
        "capacity": int(event_row["capacity"][0]),
        "user_available": is_available,
        "next_feasible_hours": round(next_feasible, 1) if next_feasible > 0 else None,
        "friends": friends_info[:20],  # Top 20 friends
        "total_intents": len(event_intents),
    }


@app.route("/")
def index():
    """Main page - user selection."""
    # Get sample users
    sample_users = state["users"].head(100).select(["user_id", "age_bucket", "schedule_type"]).to_dicts()
    return render_template("index.html", users=sample_users)


@app.route("/feed/<int:user_id>")
def feed(user_id: int):
    """Feed page for a user."""
    user_row = state["users"].filter(pl.col("user_id") == user_id)
    if len(user_row) == 0:
        return "User not found", 404

    user_info = {
        "user_id": user_id,
        "age_bucket": user_row["age_bucket"][0],
        "schedule_type": user_row["schedule_type"][0],
        "home_geo_cell": user_row["home_geo_cell"][0],
        "n_friends": len(state["friend_lookup"].get(user_id, set())),
    }

    feed_items = get_user_feed(user_id, top_k=state["cfg"].serve.top_k_feed)

    return render_template("feed.html", user=user_info, feed=feed_items)


@app.route("/event/<int:event_id>")
def event_detail(event_id: int):
    """Event detail page."""
    user_id = request.args.get("user_id", type=int, default=0)

    event_info = get_event_details(event_id, user_id)
    if not event_info:
        return "Event not found", 404

    return render_template("event.html", event=event_info, user_id=user_id)


@app.route("/api/feed/<int:user_id>")
def api_feed(user_id: int):
    """API endpoint for feed."""
    top_k = request.args.get("top_k", type=int, default=50)
    feed_items = get_user_feed(user_id, top_k=top_k)
    return jsonify(feed_items)


@app.route("/api/event/<int:event_id>")
def api_event(event_id: int):
    """API endpoint for event details."""
    user_id = request.args.get("user_id", type=int, default=0)
    event_info = get_event_details(event_id, user_id)
    return jsonify(event_info)


def main():
    parser = argparse.ArgumentParser(description="Run web server")
    parser.add_argument("--small", action="store_true", help="Use small config")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = get_config(small=args.small)

    logger.info("Loading data and models...")
    load_data(cfg)

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
