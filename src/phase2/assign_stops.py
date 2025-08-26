from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

def assign_points_to_stops(cfg, logger):
    pts = pd.read_parquet(cfg["clean_output"], columns=["latitude","longitude","user_id","datetime"])
    stops = pd.read_csv(cfg["stops_output"])
    non_depot = stops[stops.get("is_depot", 0) != 1].copy()

    if non_depot.empty:
        logger.warning("No stops to assign.")
        return

    tree = BallTree(np.radians(non_depot[["latitude","longitude"]].to_numpy()), metric="haversine")
    dist_rad, idx = tree.query(np.radians(pts[["latitude","longitude"]].to_numpy()), k=1)
    dist_m = (dist_rad[:,0] * 6371000.0)
    nearest_ids = non_depot.iloc[idx[:,0]]["stop_id"].to_numpy()

    assign_radius = float(cfg["assign_radius_m"])
    pts["assigned_stop"] = np.where(dist_m <= assign_radius, nearest_ids, -1)
    pts["dist_to_stop_m"] = dist_m

    Path(cfg["assigned_output"]).parent.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(cfg["assigned_output"], index=False)
    cov = float(np.mean(pts["assigned_stop"] >= 0))
    logger.info(f"Assigned points saved: {cfg['assigned_output']} rows={len(pts):,}, coverage={cov:.2%}")