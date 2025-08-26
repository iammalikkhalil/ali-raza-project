from __future__ import annotations
import numpy as np
import pandas as pd

from src.phase2.utils_phase2 import ll_to_m, m_to_ll

def infer_school_center(df_points: pd.DataFrame, cfg, logger):
    """Grid by ~300m, pick densest cell center as school depot."""
    grid_m = cfg["school"]["grid_cell_m"]
    x, y = ll_to_m(df_points["longitude"].to_numpy(), df_points["latitude"].to_numpy())
    gx = (x // grid_m).astype(np.int64)
    gy = (y // grid_m).astype(np.int64)
    keys, counts = np.unique(np.c_[gx, gy], axis=0, return_counts=True)
    best_idx = int(np.argmax(counts))
    cell_x = (keys[best_idx, 0] + 0.5) * grid_m
    cell_y = (keys[best_idx, 1] + 0.5) * grid_m
    lon, lat = m_to_ll(np.array([cell_x]), np.array([cell_y]))
    logger.info(f"Traffic center: {counts[best_idx]:,} pts in {grid_m}m cell → ({lat[0]:.6f}, {lon[0]:.6f})")
    return float(lat[0]), float(lon[0])