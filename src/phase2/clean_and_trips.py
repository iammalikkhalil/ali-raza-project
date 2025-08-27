from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count
from .utils_phase2 import haversine_m, kmh_from_m_dt
import time
from tqdm import tqdm  # Progress bars

def clean_and_segment_in_batches_optimized(cfg, logger):
    src = cfg.get("p2_sample_output") or cfg["phase1_parquet"]
    
    out_clean = Path(cfg["clean_output"])
    out_seg = Path(cfg["segments_output"])
    out_long = Path(cfg["long_trips_output"])
    
    # Get total rows for progress tracking
    dataset = pq.ParquetFile(src)
    total_rows = dataset.metadata.num_rows
    batch_size = 100_000  # Smaller batches for better memory management
    
    logger.info(f"Total rows to process: {total_rows:,}")
    
    all_clean, all_segs, all_long = [], [], []
    processed_rows = 0
    
    # Use tqdm for progress bar
    with tqdm(total=total_rows, desc="Processing batches") as pbar:
        for i, batch in enumerate(dataset.iter_batches(batch_size=batch_size)):
            start_time = time.time()
            
            df = batch.to_pandas()
            if df.empty:
                continue
            
            # Clean points
            df_clean = _clean_points_optimized(df, cfg)
            
            if not df_clean.empty:
                # Segment runs - optimized version
                segs = _segments_from_runs_optimized(df_clean, cfg)
                
                # Filter long trips
                long_mask = (segs["duration_sec"].between(600, 3600)) & (segs["distance_m"].between(1000, 30000))
                long_trips = segs[long_mask]
                
                all_clean.append(df_clean)
                all_segs.append(segs)
                all_long.append(long_trips)
            
            processed_rows += len(df)
            batch_time = time.time() - start_time
            
            pbar.update(len(df))
            pbar.set_postfix({
                'Batch': i,
                'Clean': f"{len(df_clean):,}",
                'Segs': f"{len(segs):,}",
                'Time': f"{batch_time:.1f}s"
            })
            
            # Clear memory
            del df, df_clean, segs, long_trips
            import gc
            gc.collect()
    
    # Save outputs
    if all_clean:
        pd.concat(all_clean, ignore_index=True).to_parquet(out_clean, index=False, compression="snappy")
    if all_segs:
        pd.concat(all_segs, ignore_index=True).to_parquet(out_seg, index=False, compression="snappy")
    if all_long:
        pd.concat(all_long, ignore_index=True).to_parquet(out_long, index=False, compression="snappy")
    
    logger.info(f"✅ Processing complete!")


def _clean_points_optimized(df: pd.DataFrame, cfg) -> pd.DataFrame:
    cont = cfg["continuity"]
    flt = cfg["filters"]
    
    # Initial filtering
    df = df.dropna(subset=["user_id", "latitude", "longitude", "datetime"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    
    # Sort and group efficiently
    df = df.sort_values(["user_id", "datetime"]).reset_index(drop=True)
    
    # Use vectorized operations with chunking
    user_groups = df.groupby("user_id")
    
    # Calculate shifts in chunks to avoid memory issues
    shifts = []
    for _, group in user_groups:
        group = group.copy()
        group["lat_prev"] = group["latitude"].shift(1)
        group["lon_prev"] = group["longitude"].shift(1)
        group["t_prev"] = group["datetime"].shift(1)
        shifts.append(group)
    
    df = pd.concat(shifts, ignore_index=True)
    
    # Vectorized calculations
    dt = (df["datetime"] - df["t_prev"]).dt.total_seconds()
    df["dt_s"] = dt
    
    # Optimized haversine - process in chunks
    def chunked_haversine(lat1, lon1, lat2, lon2, chunk_size=50000):
        distances = []
        for i in range(0, len(lat1), chunk_size):
            end = min(i + chunk_size, len(lat1))
            dist_chunk = haversine_m(
                lat1.iloc[i:end], lon1.iloc[i:end],
                lat2.iloc[i:end], lon2.iloc[i:end]
            )
            distances.extend(dist_chunk)
        return distances
    
    df["dx_m"] = chunked_haversine(df["latitude"], df["longitude"], 
                                  df["lat_prev"], df["lon_prev"])
    
    df["speed_kmh"] = kmh_from_m_dt(df["dx_m"], df["dt_s"])
    
    # Filters
    mask = (df["dt_s"] > 0) if flt.get("drop_dt_le_zero", True) else pd.Series(True, index=df.index)
    mask &= (df["speed_kmh"] <= flt["max_speed_mps"] * 3.6)
    
    # Trip breaks
    breaks = (df["dt_s"] > cont["max_gap_s"]) | (df["dx_m"] > cont["max_jump_m"]) | ~mask
    df["trip_break"] = breaks.fillna(True)
    df["trip_id"] = breaks.cumsum()
    
    return df[mask].reset_index(drop=True)

def _segments_from_runs_optimized(df: pd.DataFrame, cfg) -> pd.DataFrame:
    S = cfg["segments"]
    rows = []
    
    # Process users in smaller chunks
    user_ids = df["user_id"].unique()
    
    for user_id in tqdm(user_ids, desc="Processing users"):
        user_df = df[df["user_id"] == user_id].copy()
        if len(user_df) < 2:
            continue
            
        user_rows = _process_single_user_optimized(user_df, S)
        rows.extend(user_rows)
    
    return pd.DataFrame(rows)

def _process_single_user_optimized(user_df: pd.DataFrame, S) -> list[dict]:
    rows = []
    user_df = user_df.sort_values("datetime").reset_index(drop=True)
    
    t = user_df["datetime"].astype("int64") // 10**9
    n = len(user_df)
    start = 0
    
    while start < n - 1:
        # Find window efficiently
        t0 = t.iloc[start]
        end = start + 1
        
        # Binary search for window end
        left, right = start + 1, n - 1
        while left <= right:
            mid = (left + right) // 2
            if t.iloc[mid] - t0 < S["win_s"]:
                left = mid + 1
            else:
                right = mid - 1
        end = left
        
        if end >= n:
            break
            
        # Check duration
        dur = float(t.iloc[end - 1] - t0)
        if not (S["min_duration_s"] <= dur <= S["max_duration_s"]):
            start += 1
            continue
            
        # Check distance
        lat0, lon0 = user_df.loc[start, ["latitude", "longitude"]]
        lat1, lon1 = user_df.loc[end - 1, ["latitude", "longitude"]]
        dist = haversine_m(lat0, lon0, lat1, lon1)
        
        if not (S["min_distance_m"] <= dist <= S["max_distance_m"]):
            start += 1
            continue
            
        # Calculate metrics efficiently
        window_df = user_df.iloc[start:end]
        speeds = window_df["speed_kmh"].values
        med_speed = np.median(speeds)
        stop_ratio = np.mean(speeds < S["idle_speed_kmh"])
        
        if not (S["median_speed_kmh_min"] <= med_speed <= S["median_speed_kmh_max"]):
            start += 1
            continue
            
        if stop_ratio > S["stop_ratio_max"]:
            start += 1
            continue
            
        # Direction calculation
        dlon = np.radians(lon1 - lon0)
        latA, latB = np.radians(lat0), np.radians(lat1)
        yb = np.sin(dlon) * np.cos(latB)
        xb = np.cos(latA) * np.sin(latB) - np.sin(latA) * np.cos(latB) * np.cos(dlon)
        direction_deg = (np.degrees(np.arctan2(yb, xb)) + 360) % 360
        
        rows.append({
            "user_id": user_df["user_id"].iloc[0],
            "start_time": user_df.loc[start, "datetime"],
            "end_time": user_df.loc[end - 1, "datetime"],
            "start_hour": user_df.loc[start, "datetime"].hour,
            "weekday": user_df.loc[start, "datetime"].weekday(),
            "distance_m": dist,
            "duration_sec": dur,
            "direction_deg": direction_deg,
            "median_speed_kmh": med_speed,
            "stop_ratio": stop_ratio,
        })
        
        # Move start position
        stride_time = user_df.loc[start, "datetime"] + pd.Timedelta(seconds=S["stride_s"])
        while start < n and user_df.loc[start, "datetime"] < stride_time:
            start += 1
    
    return rows