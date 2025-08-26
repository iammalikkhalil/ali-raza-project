# src/phase1/utils_data.py
import yaml
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_config(config_path="configs/phase1_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(log_file: str, level="INFO"):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase1")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = RotatingFileHandler(log_file, maxBytes=1048576, backupCount=3)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def plot_histogram(df, column, output_path, bins=50, title=None):
    plt.figure(figsize=(6,4))
    plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(title or f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()



def save_parquet(df, output_path, engine="pyarrow", compression="snappy"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine=engine, compression=compression)

def write_summary_report(report_path, metadata, plots_paths):
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Phase 1 Summary Report\n\n")
        f.write("## Metadata\n")
        for k, v in metadata.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Plots\n")
        for desc, path in plots_paths.items():
            f.write(f"- {desc}: ![]({Path(path).as_posix()})\n")  # POSIX paths for markdown
