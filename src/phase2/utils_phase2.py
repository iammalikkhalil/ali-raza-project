from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

# ----------------- IO / logging -----------------
def load_config(path: str = "configs/phase2_config.yaml") -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str) -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase2")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger

# ----------------- Geo helpers -----------------
def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    return 2 * R * np.arcsin(np.sqrt(a))

def kmh_from_m_dt(dist_m: np.ndarray, dt_s: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (dist_m / np.maximum(dt_s, 1e-6)) * 3.6

def within_bbox(df: pd.DataFrame, bbox: Dict) -> pd.DataFrame:
    return df[(df["latitude"] >= bbox["lat_min"]) &
              (df["latitude"] <= bbox["lat_max"]) &
              (df["longitude"] >= bbox["lon_min"]) &
              (df["longitude"] <= bbox["lon_max"])]

def to_local_time(s_utc: pd.Series, tz_name: str) -> pd.Series:
    return pd.to_datetime(s_utc, utc=True, errors="coerce").dt.tz_convert(tz_name)

# Web Mercator transformers
_TO_M = Transformer.from_crs(4326, 3857, always_xy=True)
_TO_LL = Transformer.from_crs(3857, 4326, always_xy=True)

def ll_to_m(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _TO_M.transform(lon, lat)
    return np.asarray(x), np.asarray(y)

def m_to_ll(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon, lat = _TO_LL.transform(x, y)
    return np.asarray(lon), np.asarray(lat)
