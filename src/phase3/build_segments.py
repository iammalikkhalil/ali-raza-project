import pandas as pd
import numpy as np
from pathlib import Path
from src.phase2 import utils_phase2 as U  # reuse haversine_m

def make_segments(cfg, logger):
    src = cfg["gps_with_stop_id"]
    out = Path(cfg["features_parquet"])
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = ["user_id","datetime","latitude","longitude","assigned_stop"]
    df = pd.read_parquet(src, columns=cols).dropna(subset=["assigned_stop"])
    df = df[df["assigned_stop"] != -1]

    # sort
    df = df.sort_values(["user_id","datetime"]).reset_index(drop=True)

    # mark segment boundaries: when stop changes
    df["prev_stop"] = df.groupby("user_id")["assigned_stop"].shift()
    df["prev_lat"]  = df.groupby("user_id")["latitude"].shift()
    df["prev_lon"]  = df.groupby("user_id")["longitude"].shift()
    df["prev_time"] = df.groupby("user_id")["datetime"].shift()

    # keep only rows where a stop change occurred (prev_stop != assigned_stop)
    seg = df[(df["prev_stop"].notna()) & (df["assigned_stop"] != df["prev_stop"])].copy()

    # compute target + features
    seg["duration_sec"] = (seg["datetime"] - seg["prev_time"]).dt.total_seconds()
    seg["distance_m"] = U.haversine_m(seg["prev_lat"], seg["prev_lon"],
                                      seg["latitude"],  seg["longitude"])

    # speed sanity
    seg["speed_kmh"] = U.kmh_from_m_dt(seg["distance_m"], seg["duration_sec"])

    # temporal features
    seg["start_hour"] = seg["prev_time"].dt.hour + seg["prev_time"].dt.minute/60.0
    seg["weekday"]    = seg["prev_time"].dt.weekday

    # bearing (degrees)
    dlon = np.radians(seg["longitude"] - seg["prev_lon"])
    lat1 = np.radians(seg["prev_lat"]); lat2 = np.radians(seg["latitude"])
    y = np.sin(dlon)*np.cos(lat2)
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    seg["direction_deg"] = (np.degrees(np.arctan2(y, x)) + 360) % 360

    # rename stops
    seg = seg.rename(columns={"prev_stop":"start_stop", "assigned_stop":"end_stop"})

    # filter unrealistic segments
    f = cfg
    seg = seg[
        (seg["duration_sec"] >= f["min_duration_sec"]) &
        (seg["duration_sec"] <= f["max_duration_sec"]) &
        (seg["distance_m"]   >= f["min_distance_m"]) &
        (seg["distance_m"]   <= f["max_distance_m"]) &
        (seg["speed_kmh"]    <= f["max_speed_kmh"])
    ].copy()

    # select final columns
    seg = seg[[
        "user_id","start_stop","end_stop",
        "prev_lat","prev_lon","latitude","longitude",
        "start_hour","weekday","distance_m","direction_deg",
        "duration_sec"
    ]]

    seg.to_parquet(out, index=False)
    logger.info(f"Segments/features saved: {out} rows={len(seg):,}")
    return seg
