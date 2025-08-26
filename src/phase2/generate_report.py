from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _hist(path: Path, data: np.ndarray, title: str, bins=50):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(data[~np.isnan(data)], bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def build_p2_report(cfg, logger):
    # load artifacts
    pts = pd.read_parquet(cfg["assigned_output"], columns=["assigned_stop","dist_to_stop_m"])
    segs = pd.read_parquet(cfg["segments_output"]) if Path(cfg["segments_output"]).exists() else pd.DataFrame()
    stops = pd.read_csv(cfg["stops_output"])

    n_stops = int((stops.get("is_depot",0) != 1).sum())
    assigned = pts[pts["assigned_stop"] >= 0]
    coverage = float(len(assigned)) / max(1, len(pts))
    p50 = float(np.percentile(assigned["dist_to_stop_m"], 50)) if len(assigned) else float("nan")
    p90 = float(np.percentile(assigned["dist_to_stop_m"], 90)) if len(assigned) else float("nan")

    # avg stop size from assignments (more robust than kmeans size)
    if len(assigned):
        avg_size = float(assigned.groupby("assigned_stop").size().mean())
    else:
        avg_size = 0.0

    # segments quality
    if not segs.empty:
        car_like = np.mean((segs["median_speed_kmh"].between(
            cfg["segments"]["median_speed_kmh_min"], cfg["segments"]["median_speed_kmh_max"])) &
            (segs["stop_ratio"] <= cfg["segments"]["stop_ratio_max"]))
        car_like = float(car_like)
        n_segments = int(len(segs))
        med_dur = float(np.median(segs["duration_sec"]))
        med_dist = float(np.median(segs["distance_m"]))
    else:
        car_like, n_segments, med_dur, med_dist = 0.0, 0, float("nan"), float("nan")

    report = {
        "coverage_pct": coverage,
        "n_stops": n_stops,
        "avg_stop_size": avg_size,
        "dist_to_stop_p50_m": p50,
        "dist_to_stop_p90_m": p90,
        "n_segments_total": n_segments,
        "car_like_share": car_like,
        "median_segment_duration_sec": med_dur,
        "median_segment_distance_m": med_dist,
        "gates": {
            "coverage_ok": coverage >= cfg["coverage_target"],
            "p90_ok": (not np.isnan(p90)) and (p90 <= cfg["p90_distance_m_threshold"]),
            "stops_ok": (n_stops >= min(cfg["kmeans"]["k_scan"])) and (n_stops <= max(cfg["kmeans"]["k_scan"])),
        },
    }

    # figures
    fig_dir = Path(cfg["fig_dir"])
    if len(assigned):
        _hist(fig_dir / "dist_to_stop_hist.png", assigned["dist_to_stop_m"].to_numpy(), "Distance to nearest stop (m)")
    if not segs.empty:
        _hist(fig_dir / "segment_duration_hist.png", segs["duration_sec"].to_numpy(), "Segment duration (s)")
        _hist(fig_dir / "segment_distance_hist.png", segs["distance_m"].to_numpy(), "Segment distance (m)")
        _hist(fig_dir / "segment_speed_hist.png", segs["median_speed_kmh"].to_numpy(), "Median speed (km/h)")

    Path(cfg["quality_report"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["quality_report"], "w") as f:
        json.dump(report, f, indent=2)

    # log gates
    flags = []
    if not report["gates"]["coverage_ok"]: flags.append("Low coverage")
    if not report["gates"]["p90_ok"]: flags.append("High p90 distance")
    if not report["gates"]["stops_ok"]: flags.append("Stops count out of range")
    if flags:
        logger.warning(f"Quality gates flagged: {', '.join(flags)}")
    logger.info(f"Quality report: {cfg['quality_report']}")
    return report