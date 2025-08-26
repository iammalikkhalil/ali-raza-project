from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from src.phase2.utils_phase2 import haversine_m

def _coverage_for_centroids(sample_lat, sample_lon, cents_lat, cents_lon, radius_m):
    # compute nearest centroid distance haversine
    tree = BallTree(np.radians(np.c_[cents_lat, cents_lon]), metric="haversine")
    dist_rad, _ = tree.query(np.radians(np.c_[sample_lat, sample_lon]), k=1)
    dist_m = dist_rad[:,0] * 6371000.0
    cover = np.mean(dist_m <= radius_m)
    p90 = float(np.percentile(dist_m, 90))
    return cover, p90

def _kmeans_centroids(lat, lon, K, seed):
    # kmeans in radians → transform to euclidean via small-angle? Better: cluster in WebMerc? Simpler: KMeans on lat/lon scaled.
    # scale lon by cos(mean_lat) to reduce distortion
    lat_arr = np.asarray(lat); lon_arr = np.asarray(lon)
    scale = np.cos(np.radians(np.mean(lat_arr)))
    X = np.c_[lat_arr, lon_arr * scale]
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(X)
    cents = []
    for k in range(K):
        mask = (labels == k)
        if mask.sum() == 0: continue
        cents.append((float(lat_arr[mask].mean()), float(lon_arr[mask].mean())))
    if len(cents) < K:
        # empty clusters removed; KMeans can produce fewer (rare). handle upstream.
        pass
    c_lat = np.array([c[0] for c in cents]); c_lon = np.array([c[1] for c in cents])
    return c_lat, c_lon, labels

def make_bus_stops(cfg, logger):
    src = cfg.get("clean_output")
    df = pd.read_parquet(src, columns=["latitude","longitude"])
    if df.empty:
        logger.warning("No points to cluster.")
        Path(cfg["stops_output"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["stop_id","latitude","longitude","size","radius_m","is_depot"]).to_csv(cfg["stops_output"], index=False)
        return pd.DataFrame()

    # infer school location from densest cell, then keep points within radius
    from src.phase2.infer_school import infer_school_center
    lat0, lon0 = infer_school_center(df, cfg, logger)
    dist_to_school = haversine_m(df["latitude"], df["longitude"], lat0, lon0)
    df = df[dist_to_school <= cfg["school"]["radius_km"] * 1000].reset_index(drop=True)
    logger.info(f"Points within {cfg['school']['radius_km']:.1f} km of school: {len(df):,}")

    # downsample for fit
    fit_n = int(cfg["kmeans"]["downsample_points"])
    if len(df) > fit_n:
        df_fit = df.sample(n=fit_n, random_state=cfg["random_seed"]).reset_index(drop=True)
        logger.info(f"Downsampled to {len(df_fit):,} points for clustering fit.")
    else:
        df_fit = df

    # scan K to meet coverage gates
    Ks = cfg["kmeans"]["k_scan"]
    assign_r = float(cfg["assign_radius_m"])
    best = None
    for K in Ks:
        c_lat, c_lon, _ = _kmeans_centroids(df_fit["latitude"], df_fit["longitude"], K, cfg["random_seed"])
        cover, p90 = _coverage_for_centroids(df_fit["latitude"], df_fit["longitude"], c_lat, c_lon, assign_r)
        if best is None or (cover > best["cover"] or (np.isclose(cover, best["cover"]) and p90 < best["p90"])):  # max cover, then min p90
            best = {"K": K, "c_lat": c_lat, "c_lon": c_lon, "cover": cover, "p90": p90}

    # pick first K that passes both gates; else best coverage
    chosen = None
    for K in Ks:
        c_lat, c_lon, _ = _kmeans_centroids(df_fit["latitude"], df_fit["longitude"], K, cfg["random_seed"])
        cover, p90 = _coverage_for_centroids(df_fit["latitude"], df_fit["longitude"], c_lat, c_lon, assign_r)
        if cover >= cfg["coverage_target"] and p90 <= cfg["p90_distance_m_threshold"]:
            chosen = {"K": K, "c_lat": c_lat, "c_lon": c_lon, "cover": cover, "p90": p90}
            break
    if chosen is None:
        chosen = best
    logger.info(f"KMeans picked K={chosen['K']} (coverage={chosen['cover']:.2%}, p90={chosen['p90']:.1f} m)")

    # finalize stops on full in-radius set using chosen K
    c_lat, c_lon, labels = _kmeans_centroids(df["latitude"], df["longitude"], chosen["K"], cfg["random_seed"])
    df["label"] = labels
    stops = []
    for k in range(chosen["K"]):
        g = df[df["label"] == k]
        if len(g) == 0: continue
        latc, lonc = float(g["latitude"].mean()), float(g["longitude"].mean())
        rad = float(np.median(haversine_m(g["latitude"], g["longitude"], latc, lonc)))
        stops.append((k, latc, lonc, len(g), rad))
    stops_df = pd.DataFrame(stops, columns=["stop_id","latitude","longitude","size","radius_m"])

    # append depot
    depot = pd.DataFrame([{"stop_id": -1, "latitude": lat0, "longitude": lon0, "size": 0, "radius_m": 0.0, "is_depot": 1}])
    stops_df["is_depot"] = 0
    stops_df = pd.concat([stops_df, depot], ignore_index=True)

    Path(cfg["stops_output"]).parent.mkdir(parents=True, exist_ok=True)
    stops_df.to_csv(cfg["stops_output"], index=False)
    logger.info(f"Saved {len(stops_df)-1} student stops (+1 depot) → {cfg['stops_output']}")
    return stops_df