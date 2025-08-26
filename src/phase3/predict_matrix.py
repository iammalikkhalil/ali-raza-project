import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from src.phase2 import utils_phase2 as U  # for haversine


def _build_features_for_pairs(dist_m, direction_deg, weekday, hour, feature_order):
    hour_sin = np.sin(2 * np.pi * (float(hour) / 24.0))
    hour_cos = np.cos(2 * np.pi * (float(hour) / 24.0))
    theta = np.radians(direction_deg)
    dir_sin = np.sin(theta)
    dir_cos = np.cos(theta)
    distance_log = np.log1p(dist_m)

    X = pd.DataFrame({
        "start_hour": float(hour),
        "weekday": int(weekday),
        "distance_m": dist_m,
        "direction_deg": direction_deg,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "distance_log": distance_log,
        "dir_sin": dir_sin,
        "dir_cos": dir_cos,
    })
    return X[feature_order]


def _baseline_time_seconds(distance_m: np.ndarray, weekday: int, hour: float, bmap: dict) -> np.ndarray:
    hb = int(np.floor(hour)) % 24
    speed = bmap.get("by_wh", {}).get((int(weekday), hb), bmap.get("global", 8.33))
    speed = max(speed, 0.1)
    return distance_m.astype(float) / speed


def build_matrices(cfg, logger):
    stops = pd.read_csv(cfg["bus_stops_csv"])
    non_depot = stops[stops.get("is_depot", 0) != 1].copy()

    ids = non_depot["stop_id"].to_numpy()
    lat = non_depot["latitude"].to_numpy()
    lon = non_depot["longitude"].to_numpy()

    n = len(non_depot)
    src_ids = np.repeat(ids, n)
    dst_ids = np.tile(ids, n)
    lat_src = np.repeat(lat, n)
    lon_src = np.repeat(lon, n)
    lat_dst = np.tile(lat, n)
    lon_dst = np.tile(lon, n)

    # distances
    dist_m = U.haversine_m(lat_src, lon_src, lat_dst, lon_dst)
    dist_df = pd.DataFrame({"src_stop": src_ids, "dst_stop": dst_ids, "distance_m": dist_m})
    dist_df.to_csv(cfg["distance_matrix_csv"], index=False)

    # bearings
    dlon = np.radians(lon_dst - lon_src)
    lat1 = np.radians(lat_src); lat2 = np.radians(lat_dst)
    yb = np.sin(dlon) * np.cos(lat2)
    xb = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    direction_deg = (np.degrees(np.arctan2(yb, xb)) + 360) % 360

    weekday = cfg["matrix_generation"]["weekday"]
    hour = cfg["matrix_generation"]["hour_of_day"]

    bundle = joblib.load(cfg["model_path"])
    feats = bundle["features"]
    bmap = bundle["baseline_map"]

    X_df = _build_features_for_pairs(dist_m, direction_deg, weekday, hour, feats)
    base_time = _baseline_time_seconds(dist_m, weekday, hour, bmap)

    # point prediction (log ratio)
    point = bundle.get("point_model", bundle.get("model"))
    pred_logratio = point.predict(X_df)
    time_p50 = base_time * np.exp(pred_logratio)

    time_df = dist_df.copy()
    time_df["pred_time_sec"] = time_p50
    time_df.to_csv(cfg["time_matrix_csv"], index=False)

    # quantiles (ratio directly)
    q_models = bundle.get("quantile_models", {})
    if "0.9" in q_models:
        r90 = q_models["0.9"].predict(X_df)  # ratio
        out90 = cfg["time_matrix_csv"].replace(".csv", "_p90.csv")
        time_df90 = dist_df.copy()
        time_df90["pred_time_sec"] = base_time * r90
        time_df90.to_csv(out90, index=False)
        logger.info(f"Time matrix (p90): {out90}")

    pivot = time_df.pivot(index="src_stop", columns="dst_stop", values="pred_time_sec")
    Path(cfg["time_matrix_pivot"]).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_parquet(cfg["time_matrix_pivot"])

    logger.info(f"Distance matrix: {cfg['distance_matrix_csv']}")
    logger.info(f"Time matrix (p50): {cfg['time_matrix_csv']}")
    logger.info(f"Time matrix (pivot p50): {cfg['time_matrix_pivot']}")
    return {"n_pairs": int(len(time_df))}