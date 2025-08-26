from __future__ import annotations 
from pathlib import Path 
import pyarrow.dataset as ds 
import pandas as pd 

from src.phase2.utils_phase2 import within_bbox, to_local_time 

def make_p2_sample(cfg, logger): 
    src = cfg["phase1_parquet"] 
    out = Path(cfg["p2_sample_output"]) 
    out.parent.mkdir(parents=True, exist_ok=True) 

    target = int(cfg["target_sample_rows"]) 
    seed = int(cfg.get("shuffle_seed", 42)) 
    bbox = cfg["bbox"] 
    mw = cfg["morning_window"] 

    dataset = ds.dataset(src, format="parquet") 
    cols = [c for c in ["user_id","latitude","longitude","altitude","datetime","date","time"] if c in {f.name for f in dataset.schema}] 
    buf = []; total = 0 

    for batch in dataset.to_batches(columns=cols, batch_size=200_000): 
        df = batch.to_pandas() 
        df = within_bbox(df, bbox) 

        if mw.get("enable", False): 
            dt_local = to_local_time(df["datetime"], cfg["timezone"]) 
            hhmm = dt_local.dt.strftime("%H:%M") 
            df = df[(hhmm >= mw["start"]) & (hhmm <= mw["end"])] 

        if df.empty: 
            continue 
        buf.append(df) 
        total += len(df) 
        if total >= target * 1.3: 
            break 
                 
    if not buf: 
        logger.error("No rows after bbox/time filters. Check config.") 
        return 
         
    big = pd.concat(buf, ignore_index=True) 
    if len(big) > target: 
        big = big.sample(n=target, random_state=seed).reset_index(drop=True) 
           
    big.to_parquet(out, index=False) 
    logger.info(f"Sample written: {out} rows={len(big):,}") 