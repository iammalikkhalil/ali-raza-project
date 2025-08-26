import pandas as pd
import numpy as np
from pathlib import Path
import hdbscan 
from sklearn.neighbors import BallTree
import json
import logging

EARTH_RADIUS_M = 6371000.0


# ----------------------------
# Utils
# ----------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in meters."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * (2 * np.arcsin(np.sqrt(a)))


def setup_logger(log_file="reports/phase2_full.log"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase2_full")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


# ----------------------------
# Step 1: Clean + Segment (streaming)
# ----------------------------
def segment_and_save_chunks(src, out_path, chunk_size=500_000, logger=None):
    """
    Process huge GPS data in chunks, write to cleaned parquet.
    Returns: cleaned parquet path.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0

    for i, chunk in enumerate(pd.read_parquet(src, chunksize=chunk_size, engine="pyarrow")):
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"])
        chunk = chunk.sort_values(["user_id", "timestamp"])

        # trip segmentation
        chunk["lat_prev"] = chunk.groupby("user_id")["latitude"].shift()
        chunk["lon_prev"] = chunk.groupby("user_id")["longitude"].shift()
        chunk["t_prev"] = chunk.groupby("user_id")["timestamp"].shift()

        chunk["dist_m"] = haversine_m(chunk["lat_prev"], chunk["lon_prev"],
                                      chunk["latitude"], chunk["longitude"])
        chunk["dt_s"] = (chunk["timestamp"] - chunk["t_prev"]).dt.total_seconds()
              
        new_trip = (chunk["dt_s"] > 600) | (chunk["dist_m"] > 2000)
        chunk["trip_id"] = new_trip.cumsum()
              
        # filter realistic
        chunk = chunk[(chunk["dt_s"] > 0) & (chunk["dt_s"] < 3600) & (chunk["dist_m"] < 30000)]
            
        # write append
        if writer is None:
            writer = pd.DataFrame(chunk).to_parquet(out_path, index=False, engine="pyarrow")
        else:
            chunk.to_parquet(out_path, index=False, engine="pyarrow", append=True)
             
        total_rows += len(chunk)
        if logger:
            logger.info(f"Processed chunk {i}, rows kept={len(chunk):,}")
         
    return out_path, total_rows
        
         
# ----------------------------
# Step 2: Cluster stops
# ----------------------------
def cluster_stops(cleaned_parquet, max_stops=250, logger=None):
    df = pd.read_parquet(cleaned_parquet, columns=["latitude", "longitude"])
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())

    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, metric="haversine", core_dist_n_jobs=-1)
    labels = clusterer.fit_predict(coords)
    df["stop_id"] = labels

    stops = df.groupby("stop_id").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        size=("stop_id", "size")
    ).reset_index()

    stops = stops[stops["stop_id"] >= 0]
    stops = stops.sort_values("size", ascending=False).head(max_stops)

    depot_id = stops.iloc[0]["stop_id"]
    stops["is_depot"] = (stops["stop_id"] == depot_id).astype(int)

    if logger:
        logger.info(f"Clustered {len(stops)} stops (depot={depot_id})")
    return stops, df


# ----------------------------
# Step 3: Segments
# ----------------------------
def build_segments(df):
    segs = []
    for tid, trip in df.groupby("trip_id"):
        trip = trip.sort_values("timestamp")
        stops = trip["stop_id"].to_numpy()
        times = trip["timestamp"].to_numpy()
        lats = trip["latitude"].to_numpy()
        lons = trip["longitude"].to_numpy()

        for i in range(len(stops) - 1):
            if stops[i] < 0 or stops[i + 1] < 0 or stops[i] == stops[i + 1]:
                continue
            dur = (times[i + 1] - times[i]).astype("timedelta64[s]").astype(int)
            dist = haversine_m(lats[i], lons[i], lats[i + 1], lons[i + 1])
            if 30 <= dur <= 3600 and 100 <= dist <= 30000:
                segs.append([stops[i], stops[i + 1], dur, dist, trip["user_id"].iloc[0]])
    return pd.DataFrame(segs, columns=["src_stop", "dst_stop", "duration_sec", "distance_m", "user_id"])


# ----------------------------
# Step 4: Quality
# ----------------------------
def quality_report(df, stops, segs, out_path):
    coverage = (df["stop_id"] >= 0).mean()
    p50 = np.percentile(df["dist_m"].dropna(), 50)
    p90 = np.percentile(df["dist_m"].dropna(), 90)
        
    report = {
        "coverage_pct": float(coverage),
        "n_stops": int(len(stops)),
        "avg_stop_size": float(stops["size"].mean()),
        "dist_to_stop_p50_m": float(p50),
        "dist_to_stop_p90_m": float(p90),
        "n_segments_total": int(len(segs)),
        "median_segment_duration_sec": float(segs["duration_sec"].median()),
        "median_segment_distance_m": float(segs["distance_m"].median()),
    }        
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ----------------------------
# Main
# ----------------------------
def run_phase2_full(cfg):
    logger = setup_logger(cfg.get("log_file", "reports/phase2_full.log"))

    # 1. Segment (streaming)
    logger.info("Segmenting trips in chunks...")
    cleaned_parquet, total = segment_and_save_chunks(cfg["raw_gps"], cfg["gps_with_stop_id"], logger=logger)
    logger.info(f"Cleaned + segmented rows total={total:,}")

    # 2. Cluster stops
    logger.info("Clustering stops...")
    stops, df = cluster_stops(cleaned_parquet, max_stops=250, logger=logger)

    # 3. Segments
    logger.info("Building segments...")
    segs = build_segments(df)

    # 4. Save
    stops.to_csv(cfg["bus_stops_csv"], index=False)
    segs.to_parquet(cfg["segments_parquet"], index=False)

    # 5. Quality
    report = quality_report(df, stops, segs, cfg["quality_report"])
    logger.info(f"Phase 2 full done: {report}")
    return report


if __name__ == "__main__":
    cfg = {
        "raw_gps": "data/raw/beijing_all.parquet",
        "bus_stops_csv": "data/processed/bus_stops.csv",
        "gps_with_stop_id": "data/processed/gps_with_stop_id.parquet",
        "segments_parquet": "data/processed/segments.parquet",
        "quality_report": "data/processed/p2_quality_report.json",
        "log_file": "reports/phase2_full.log"
    }   
    run_phase2_full(cfg)
           
           
           