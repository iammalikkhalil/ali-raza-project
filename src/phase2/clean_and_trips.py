from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool, cpu_count

from src.phase2.utils_phase2 import haversine_m, kmh_from_m_dt


# ------------------ CLEAN ------------------
def _clean_points(df: pd.DataFrame, cfg) -> pd.DataFrame:
    cont = cfg["continuity"]
    flt = cfg["filters"]

    df = df.dropna(subset=["user_id", "latitude", "longitude", "datetime"]).copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Vectorized prev-shifts within user_id
    df = df.sort_values(["user_id", "datetime"]).reset_index(drop=True)
    df["lat_prev"] = df.groupby("user_id")["latitude"].shift(1)
    df["lon_prev"] = df.groupby("user_id")["longitude"].shift(1)
    df["t_prev"] = df.groupby("user_id")["datetime"].shift(1)

    dt = (df["datetime"] - df["t_prev"]).dt.total_seconds()
    df["dt_s"] = dt
    dx = haversine_m(df["latitude"], df["longitude"], df["lat_prev"], df["lon_prev"])
    df["dx_m"] = dx
    df["speed_kmh"] = kmh_from_m_dt(dx, dt)

    # Filters
    bad = (df["dt_s"] <= 0) if flt.get("drop_dt_le_zero", True) else np.zeros(len(df), dtype=bool)
    bad |= (df["speed_kmh"] > flt["max_speed_mps"] * 3.6)

    # Trip breaks
    breaks = (df["dt_s"] > cont["max_gap_s"]) | (df["dx_m"] > cont["max_jump_m"]) | bad
    df["trip_break"] = breaks.fillna(True)
    df["trip_id"] = breaks.cumsum()

    return df[~bad].reset_index(drop=True)


# ------------------ SEGMENTS ------------------
def _segments_from_runs_for_user(group: pd.DataFrame, S) -> list[dict]:
    rows = []
    trip = group.sort_values("datetime").reset_index(drop=True)
    if len(trip) < 2:
        return rows

    t = trip["datetime"].astype("int64") // 10**9
    start = 0
    while start < len(trip) - 1:
        t0 = t.iloc[start]
        end = start + 1
        while end < len(trip) and (t.iloc[end] - t0) < S["win_s"]:
            end += 1
        if end >= len(trip):
            break

        dur = float(t.iloc[end - 1] - t0)
        if dur < S["min_duration_s"] or dur > S["max_duration_s"]:
            start += 1
            continue

        lat0, lon0 = trip.loc[start, ["latitude", "longitude"]]
        lat1, lon1 = trip.loc[end - 1, ["latitude", "longitude"]]
        dist = float(haversine_m(lat0, lon0, lat1, lon1))
        if dist < S["min_distance_m"] or dist > S["max_distance_m"]:
            start += 1
            continue

        med_speed = float(np.median(trip.loc[start:end, "speed_kmh"]))
        stop_ratio = float(np.mean((trip.loc[start:end, "speed_kmh"] < S["idle_speed_kmh"])))
        if not (S["median_speed_kmh_min"] <= med_speed <= S["median_speed_kmh_max"]):
            start += 1; continue
        if stop_ratio > S["stop_ratio_max"]:
            start += 1; continue

        # direction
        dlon = np.radians(lon1 - lon0)
        latA = np.radians(lat0); latB = np.radians(lat1)
        yb = np.sin(dlon) * np.cos(latB)
        xb = np.cos(latA) * np.sin(latB) - np.sin(latA) * np.cos(latB) * np.cos(dlon)
        direction_deg = (np.degrees(np.arctan2(yb, xb)) + 360) % 360

        rows.append({
            "user_id": trip["user_id"].iloc[0],
            "start_time": trip.loc[start, "datetime"],
            "end_time": trip.loc[end - 1, "datetime"],
            "start_hour": trip.loc[start, "datetime"].hour + trip.loc[start, "datetime"].minute/60.0,
            "weekday": trip.loc[start, "datetime"].weekday(),
            "distance_m": dist,
            "duration_sec": dur,
            "direction_deg": direction_deg,
            "median_speed_kmh": med_speed,
            "stop_ratio": stop_ratio,
        })

        # stride
        stride_end_time = trip.loc[start, "datetime"] + pd.to_timedelta(S["stride_s"], unit="s")
        while start < len(trip) and trip.loc[start, "datetime"] < stride_end_time:
            start += 1

    return rows


def _segments_from_runs(df: pd.DataFrame, cfg) -> pd.DataFrame:
    S = cfg["segments"]
    # Parallelize across user_id
    with Pool(max(1, cpu_count()-1)) as pool:
        results = pool.starmap(_segments_from_runs_for_user, [(g, S) for _, g in df.groupby("user_id")])
    rows = [row for sub in results for row in sub]
    return pd.DataFrame(rows)


# ------------------ MAIN ------------------
def clean_and_segment_in_batches(cfg, logger):
    src = cfg.get("p2_sample_output") or cfg["phase1_parquet"]

    out_clean = Path(cfg["clean_output"])
    out_seg = Path(cfg["segments_output"])
    out_long = Path(cfg["long_trips_output"])

    dataset = ds.dataset(src, format="parquet")
    batch_size = 500_000

    all_clean = []
    all_segs = []
    all_long = []

    for i, batch in enumerate(dataset.to_batches(batch_size=batch_size)):
        df = batch.to_pandas()
        if df.empty: 
            continue

        df_clean = _clean_points(df, cfg)
        if df_clean.empty:
            continue

        segs = _segments_from_runs(df_clean, cfg)
        long_mask = (segs["duration_sec"].between(600, 3600)) & (segs["distance_m"].between(100, 30000))

        all_clean.append(df_clean)
        all_segs.append(segs)
        all_long.append(segs[long_mask])

        logger.info(f"Batch {i} → clean={len(df_clean):,}, segs={len(segs):,}")

    # Single output files
    if all_clean:
        pd.concat(all_clean, ignore_index=True).to_parquet(out_clean, index=False, compression="snappy")
    if all_segs:
        pd.concat(all_segs, ignore_index=True).to_parquet(out_seg, index=False, compression="snappy")
    if all_long:
        pd.concat(all_long, ignore_index=True).to_parquet(out_long, index=False, compression="snappy")

    logger.info(f"✅ Done. Outputs:\n  Clean={out_clean}\n  Segments={out_seg}\n  Long={out_long}")
