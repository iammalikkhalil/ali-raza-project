from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans  # Faster than regular KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm  # Progress bars
import time

from src.phase2.utils_phase2 import haversine_m

def _coverage_for_centroids(sample_lat, sample_lon, cents_lat, cents_lon, radius_m):
    """Optimized coverage calculation"""
    tree = BallTree(np.radians(np.c_[cents_lat, cents_lon]), metric="haversine")
    dist_rad, _ = tree.query(np.radians(np.c_[sample_lat, sample_lon]), k=1)
    dist_m = dist_rad[:, 0] * 6371000.0
    cover = np.mean(dist_m <= radius_m)
    p90 = float(np.percentile(dist_m, 90))
    return cover, p90

def _fast_kmeans_centroids(lat, lon, K, seed):
    """Optimized KMeans with MiniBatch and better scaling"""
    lat_arr = np.asarray(lat)
    lon_arr = np.asarray(lon)
    
    # Better scaling to account for Earth's curvature
    mean_lat = np.mean(lat_arr)
    scale_lon = np.cos(np.radians(mean_lat))
    
    # Scale coordinates
    X = np.c_[lat_arr, lon_arr * scale_lon]
    
    # Use MiniBatchKMeans for faster clustering
    km = MiniBatchKMeans(
        n_clusters=K, 
        n_init=3,  # Reduced from 10
        random_state=seed,
        batch_size=1024,  # Process in batches
        max_iter=50,  # Reduced iterations
        compute_labels=True
    )
    
    labels = km.fit_predict(X)
    cents = km.cluster_centers_
    
    # Convert back to original coordinates
    c_lat = cents[:, 0]
    c_lon = cents[:, 1] / scale_lon
    
    return c_lat, c_lon, labels

def make_bus_stops_optimized(cfg, logger):
    """Optimized version with progress tracking"""
    start_time = time.time()
    
    src = cfg.get("clean_output")
    logger.info(f"Loading data from {src}...")
    
    # Load only necessary columns to save memory
    df = pd.read_parquet(src, columns=["latitude", "longitude"])
    if df.empty:
        logger.warning("No points to cluster.")
        Path(cfg["stops_output"]).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["stop_id","latitude","longitude","size","radius_m","is_depot"]).to_csv(cfg["stops_output"], index=False)
        return pd.DataFrame()

    # Infer school location
    from src.phase2.infer_school import infer_school_center
    lat0, lon0 = infer_school_center(df, cfg, logger)
    
    # Filter points near school
    logger.info("Filtering points near school...")
    dist_to_school = haversine_m(df["latitude"], df["longitude"], lat0, lon0)
    df = df[dist_to_school <= cfg["school"]["radius_km"] * 1000].reset_index(drop=True)
    logger.info(f"Points within {cfg['school']['radius_km']:.1f} km of school: {len(df):,}")

    # Downsample for fitting
    fit_n = int(cfg["kmeans"]["downsample_points"])
    if len(df) > fit_n:
        df_fit = df.sample(n=fit_n, random_state=cfg["random_seed"]).reset_index(drop=True)
        logger.info(f"Downsampled to {len(df_fit):,} points for clustering fit.")
    else:
        df_fit = df

    # Scan K values with progress tracking
    Ks = cfg["kmeans"]["k_scan"]
    assign_r = float(cfg["assign_radius_m"])
    
    best = None
    results = []
    
    logger.info(f"Scanning K values: {Ks}")
    for K in tqdm(Ks, desc="Testing K values"):
        c_lat, c_lon, _ = _fast_kmeans_centroids(
            df_fit["latitude"], 
            df_fit["longitude"], 
            K, 
            cfg["random_seed"]
        )
        
        cover, p90 = _coverage_for_centroids(
            df_fit["latitude"], 
            df_fit["longitude"], 
            c_lat, 
            c_lon, 
            assign_r
        )
        
        results.append({"K": K, "cover": cover, "p90": p90})
        
        if best is None or (cover > best["cover"] or 
                           (np.isclose(cover, best["cover"]) and p90 < best["p90"])):
            best = {"K": K, "c_lat": c_lat, "c_lon": c_lon, "cover": cover, "p90": p90}

    # Choose the best K that meets criteria
    chosen = None
    for result in results:
        if (result["cover"] >= cfg["coverage_target"] and 
            result["p90"] <= cfg["p90_distance_m_threshold"]):
            chosen = result
            break
    
    if chosen is None:
        chosen = best
        logger.warning(f"No K met both criteria. Using best: K={chosen['K']}")
    
    logger.info(f"Selected K={chosen['K']} (coverage={chosen['cover']:.2%}, p90={chosen['p90']:.1f} m)")

    # Final clustering on full dataset
    logger.info("Performing final clustering on full dataset...")
    c_lat, c_lon, labels = _fast_kmeans_centroids(
        df["latitude"], 
        df["longitude"], 
        chosen["K"], 
        cfg["random_seed"]
    )
    
    df["label"] = labels
    
    # Create stops dataframe
    stops = []
    unique_labels = np.unique(labels)
    
    for k in tqdm(unique_labels, desc="Creating stops"):
        g = df[df["label"] == k]
        if len(g) == 0:
            continue
            
        latc, lonc = float(g["latitude"].mean()), float(g["longitude"].mean())
        
        # Faster radius calculation using sample
        if len(g) > 1000:
            sample_g = g.sample(n=1000, random_state=cfg["random_seed"])
        else:
            sample_g = g
            
        distances = haversine_m(sample_g["latitude"], sample_g["longitude"], latc, lonc)
        rad = float(np.median(distances))
        
        stops.append((k, latc, lonc, len(g), rad))

    stops_df = pd.DataFrame(stops, columns=["stop_id", "latitude", "longitude", "size", "radius_m"])

    # Add depot
    depot = pd.DataFrame([{
        "stop_id": -1, 
        "latitude": lat0, 
        "longitude": lon0, 
        "size": 0, 
        "radius_m": 0.0, 
        "is_depot": 1
    }])
    
    stops_df["is_depot"] = 0
    stops_df = pd.concat([stops_df, depot], ignore_index=True)

    # Save results
    Path(cfg["stops_output"]).parent.mkdir(parents=True, exist_ok=True)
    stops_df.to_csv(cfg["stops_output"], index=False)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Saved {len(stops_df)-1} student stops (+1 depot) in {total_time:.1f} seconds")
    logger.info(f"→ {cfg['stops_output']}")
    
    return stops_df

# Replace your original function with this optimized version
make_bus_stops = make_bus_stops_optimized