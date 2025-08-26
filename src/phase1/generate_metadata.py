from pathlib import Path
import json
import math
from typing import List, Dict, Tuple, Iterable

import pandas as pd
import numpy as np

from src.phase1.utils_data import (
    load_config,
    setup_logging,
    plot_histogram,
    write_summary_report,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    raise RuntimeError("Install pyarrow to write parquet efficiently.") from e


def _read_plt_to_df(path: Path, expected_columns: List[str]) -> pd.DataFrame:
    dtypes = {
        "latitude": "float64",
        "longitude": "float64",
        "unused": "float64",
        "altitude": "float64",
        "days": "float64",
        "date": "string",
        "time": "string",
    }
    df = pd.read_csv(
        path,
        skiprows=6,
        header=None,
        names=expected_columns,
        dtype=dtypes,
        on_bad_lines="skip",
        encoding_errors="ignore",
    )

    # altitude cleanup
    if "altitude" in df.columns:
        df.loc[df["altitude"] == -777, "altitude"] = np.nan

    # combine date + time → UTC datetime (Geolife provides GMT strings)
    if {"date", "time"}.issubset(df.columns):
        dt = (df["date"].astype("string").str.strip() + " " +
              df["time"].astype("string").str.strip()).str.strip()
        df["datetime"] = pd.to_datetime(dt, utc=True, errors="coerce")

    return df


def _write_stream_to_parquet(
    frames: Iterable[pd.DataFrame],
    out_path: Path,
    logger,
    compression: str = "snappy",
) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    total_rows = 0
    total_groups = 0

    for i, df in enumerate(frames):
        if df.empty:
            continue
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
        writer.write_table(table)
        total_rows += len(df)
        total_groups += 1
        if (i + 1) % 200 == 0:
            logger.info(f"  wrote {i+1} chunks; rows so far={total_rows:,}")

    if writer is not None:
        writer.close()
    return total_rows, total_groups


def build_phase1_artifacts(valid_files: List[Path]):
    """
    Streams valid PLT files → single parquet
    Writes metadata.json
    Creates small-sample histograms
    Writes summary_report.md
    """
    config = load_config()
    logger = setup_logging(config["logging"]["log_file"], config["logging"]["log_level"])

    expected_cols = config["expected_columns"]
    processed_dir = Path(config["processed_data_dir"])
    figures_dir = Path(config["figures_dir"])
    summary_report = Path(config["summary_report"])
    metadata_path = Path(config["metadata_output"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating metadata for {len(valid_files)} files")

    # def frames():
    #     for f in valid_files:
    #         try:
    #             df = _read_plt_to_df(f, expected_cols)
    #             # .../<user_id>/Trajectory/file.plt
    #             df["user_id"] = f.parent.parent.name
    #             yield df
    #         except Exception as e:
    #             logger.exception(f"Failed to read {f}: {e}")

    def frames():
        for f in valid_files:
            try:
                df = _read_plt_to_df(f, expected_cols)
                # .../<user_id>/Trajectory/file.plt
                uid = f.parent.parent.name
                df["user_id"] = uid

                # Try to store user_id as integer if folder name is numeric (nullable Int64)
                uid_num = pd.to_numeric(df["user_id"], errors="coerce")
                if not uid_num.isna().all():
                    df["user_id"] = uid_num.astype("Int64")

                # Reorder columns so user_id is present and consistent in output
                ordered_cols = expected_cols + ["datetime", "user_id"]
                df = df[[c for c in ordered_cols if c in df.columns]]

                yield df
            except Exception as e:
                logger.exception(f"Failed to read {f}: {e}")

    combined_path = processed_dir / "phase1_combined.parquet"
    total_rows, total_groups = _write_stream_to_parquet(frames(), combined_path, logger, compression="snappy")
    logger.info(f"Combined dataset saved to {combined_path} (rows={total_rows:,}, row_groups={total_groups})")

    # metadata
    user_ids = {f.parent.parent.name for f in valid_files}
    metadata: Dict[str, object] = {
        "num_files": len(valid_files),
        "num_points": total_rows,
        "num_users": len(user_ids),
        "columns": expected_cols + ["datetime", "user_id"],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as fp:
        json.dump(metadata, fp, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    # lightweight EDA on a sample (don’t load full parquet back)
    sample_frames = []
    if valid_files:
        sample_n = min(50, len(valid_files))
        stride = max(1, len(valid_files) // sample_n)
        for p in valid_files[::stride]:
            try:
                sample_frames.append(_read_plt_to_df(p, expected_cols)[["latitude", "longitude", "altitude"]])
            except Exception:
                pass
    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()

    plots = {}
    if not sample_df.empty:
        lat_path = figures_dir / "latitude_hist.png"
        plot_histogram(sample_df, "latitude", lat_path, title="Latitude Distribution")
        plots["Latitude Distribution"] = lat_path

        lon_path = figures_dir / "longitude_hist.png"
        plot_histogram(sample_df, "longitude", lon_path, title="Longitude Distribution")
        plots["Longitude Distribution"] = lon_path

        alt_path = figures_dir / "altitude_hist.png"
        plot_histogram(sample_df, "altitude", alt_path, title="Altitude Distribution")
        plots["Altitude Distribution"] = alt_path

    write_summary_report(summary_report, metadata, plots)
    logger.info(f"Summary report generated at {summary_report}")

    return combined_path, metadata
