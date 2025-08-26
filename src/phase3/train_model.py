import json
import warnings
from math import pi
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor


# ------------------ utils ------------------
def _import_lightgbm(logger):
    try:
        import lightgbm as lgb  # type: ignore
        return lgb
    except Exception as e:
        logger.warning(f"LightGBM not available ({e}). Falling back to RandomForest.")
        return None


def _build_feature_frame(df: pd.DataFrame, add_cyc: bool):
    """Stable, VRP-friendly features. Do not depend on Phase-2 internals."""
    need = ["start_hour", "weekday", "distance_m", "direction_deg"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Required feature '{c}' not found in features parquet.")
    X = df[need].copy()

    # cyclical time
    if add_cyc:
        rad = 2 * pi * (X["start_hour"].astype(float) / 24.0)
        X["hour_sin"] = np.sin(rad)
        X["hour_cos"] = np.cos(rad)

    # distance log
    X["distance_log"] = np.log1p(X["distance_m"].astype(float))

    # heading cycles
    theta = np.radians(X["direction_deg"].astype(float))
    X["dir_sin"] = np.sin(theta)
    X["dir_cos"] = np.cos(theta)

    return X, list(X.columns)


def _barplot_metrics(path: Path, mae: float, rmse: float, mae_baseline=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["MAE (s)", "RMSE (s)"]
    vals = [mae, rmse]
    if mae_baseline is not None:
        labels.append("MAE baseline (s)")
        vals.append(mae_baseline)
    plt.figure()
    plt.bar(labels, vals)
    plt.title("Phase 3: Test Metrics")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _adaptive_user_split(groups: np.ndarray, seed: int, test_size: float):
    rng = np.random.RandomState(seed)
    users = np.unique(groups)
    rng.shuffle(users)
    n_users = len(users)
    n_test = max(1, int(round(n_users * test_size)))
    if n_users - n_test < 2:
        n_test = max(1, n_users - 2)
    test_users = set(users[:n_test])
    mask_test = np.array([g in test_users for g in groups])
    return mask_test, users, n_users


def _filter_params(params: dict, allowed_keys: set, defaults: dict) -> dict:
    out = {k: v for k, v in (params or {}).items() if k in allowed_keys}
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out


# ---------- baseline speed (robust) ----------
def _build_baseline_speed_map(df: pd.DataFrame, max_speed_kmh: float) -> dict:
    """Return dict with:
       - 'by_wh' : {(weekday,hour_bin): speed_mps}
       - 'global': global_mps
    Speeds computed from valid segments only, using medians (robust)."""
    # compute speed
    mps = df["distance_m"].astype(float) / np.maximum(df["duration_sec"].astype(float), 1.0)
    kmh = mps * 3.6
    valid = (kmh > 1.0) & (kmh <= float(max_speed_kmh))
    dfv = df.loc[valid].copy()
    if dfv.empty:
        # pathological fallback
        global_mps = 8.33  # ~30 km/h
        return {"by_wh": {}, "global": float(global_mps)}

    dfv["speed_mps"] = dfv["distance_m"].astype(float) / np.maximum(dfv["duration_sec"].astype(float), 1.0)
    # hour bin
    dfv["hour_bin"] = np.floor(dfv["start_hour"].astype(float)).astype(int).clip(0, 23)

    # median per (weekday, hour_bin)
    by = (
        dfv.groupby(["weekday", "hour_bin"])["speed_mps"]
        .median()
        .reset_index()
    )
    baseline = { (int(r.weekday), int(r.hour_bin)) : float(r.speed_mps) for r in by.itertuples(index=False) }

    # global
    global_mps = float(dfv["speed_mps"].median())
    return {"by_wh": baseline, "global": global_mps}


def _baseline_time_seconds(distance_m: np.ndarray, weekday: np.ndarray, start_hour: np.ndarray, bmap: dict) -> np.ndarray:
    hb = np.floor(start_hour.astype(float)).astype(int).clip(0, 23)
    speeds = np.full(len(distance_m), bmap["global"], dtype=float)
    by_wh = bmap.get("by_wh", {})
    for i in range(len(distance_m)):
        key = (int(weekday[i]), int(hb[i]))
        if key in by_wh:
            speeds[i] = by_wh[key]
    speeds = np.maximum(speeds, 0.1)  # avoid zeros
    return distance_m.astype(float) / speeds


# ---------- CV that evaluates on seconds (ratio model) ----------
def _cv_mae_seconds_ratio(model_ctor, ctor_params, X: pd.DataFrame, base_time: np.ndarray,
                          y_logratio: np.ndarray, y_sec: np.ndarray, groups: np.ndarray, folds: int):
    cv_mae_sec = []
    n_groups = len(np.unique(groups))
    n_splits = min(folds, n_groups)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr_idx, va_idx in gkf.split(X, y_sec, groups=groups):
            m = model_ctor(**ctor_params)
            m.fit(X.iloc[tr_idx], y_logratio[tr_idx])
            pred_logratio = m.predict(X.iloc[va_idx])
            pred_time = base_time[va_idx] * np.exp(pred_logratio)
            cv_mae_sec.append(mean_absolute_error(y_sec[va_idx], pred_time))
    return cv_mae_sec, n_splits


# ------------------ trainer ------------------
def train_phase3(cfg, logger):
    """
    Deterministic, problem-first training:
      1) Build robust baseline speed(weekday, hour).
      2) Model multiplicative correction: log(time_ratio).
      3) Evaluate vs baseline and require improvement.

    Bundle includes the baseline map to reuse in matrix prediction.
    """
    feats_path = cfg["features_parquet"]
    model_path = Path(cfg["model_path"])
    report_path = Path(cfg["eval_report"])
    fig_dir = Path(cfg.get("fig_dir", "reports/phase3/figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load
    if not Path(feats_path).exists():
        raise FileNotFoundError(f"Features parquet not found: {feats_path}")
    df = pd.read_parquet(feats_path)
    if df.empty:
        raise ValueError("Features dataframe is empty.")
    for col in ["user_id", "duration_sec", "distance_m", "start_hour", "weekday", "direction_deg"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' in features parquet.")

    # Clean & features
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["user_id", "duration_sec", "distance_m", "start_hour", "weekday", "direction_deg"]
    )
    add_cyc = bool(cfg.get("features", {}).get("add_cyclical_time", True))
    X_df, feature_names = _build_feature_frame(df, add_cyc)

    y_sec = df["duration_sec"].astype(float).to_numpy()
    dist_m = df["distance_m"].astype(float).to_numpy()
    weekday = df["weekday"].astype(int).to_numpy()
    start_hour = df["start_hour"].astype(float).to_numpy()
    groups = df["user_id"].astype(str).to_numpy()

    # Baseline map + baseline time
    max_speed_kmh = float(cfg.get("max_speed_kmh", 130))
    bmap = _build_baseline_speed_map(df, max_speed_kmh=max_speed_kmh)
    base_time = _baseline_time_seconds(dist_m, weekday, start_hour, bmap)

    # Ratio target in log space
    # guard tiny/negative base_time
    base_time = np.maximum(base_time, 1.0)
    ratio = y_sec / base_time
    ratio = np.clip(ratio, 1e-3, 1e3)
    y_logratio = np.log(ratio)

    # Split by user
    seed = int(cfg.get("random_seed", 42))
    mask_test, users, n_users = _adaptive_user_split(groups, seed, float(cfg.get("test_size", 0.2)))
    X_train, X_test = X_df[~mask_test], X_df[mask_test]
    base_train, base_test = base_time[~mask_test], base_time[mask_test]
    y_train_logr, y_test_logr = y_logratio[~mask_test], y_logratio[mask_test]
    y_train_sec,  y_test_sec  = y_sec[~mask_test],  y_sec[mask_test]
    groups_train = groups[~mask_test]
    if len(X_train) == 0:
        raise ValueError("No training samples after user-based split.")

    # Model choice
    model_cfg = cfg.get("model", {}) or {}
    mtype = (model_cfg.get("type", "lightgbm") or "lightgbm").lower()
    raw_params = dict(model_cfg.get("params", {}))
    desired_folds = int(cfg.get("cv_folds", 3))

    allowed_rf = {
        "n_estimators","criterion","max_depth","min_samples_split","min_samples_leaf",
        "min_weight_fraction_leaf","max_features","max_leaf_nodes","min_impurity_decrease",
        "bootstrap","oob_score","n_jobs","random_state","verbose","warm_start",
        "ccp_alpha","max_samples"
    }
    rf_defaults = {"n_estimators": 300, "n_jobs": -1}

    allowed_lgb = {
        "n_estimators","learning_rate","num_leaves","max_depth","subsample",
        "colsample_bytree","reg_alpha","reg_lambda","min_child_samples",
        "min_child_weight","subsample_freq","objective","n_jobs","random_state",
        "verbose","metric"
    }
    lgb_defaults = {"n_estimators": 800, "n_jobs": -1, "verbose": -1, "metric": "mae", "objective": "regression"}

    lgb = _import_lightgbm(logger) if mtype == "lightgbm" else None
    use_lgb = (mtype == "lightgbm" and lgb is not None)

    # Train (+ CV on seconds)
    if use_lgb:
        point_params = _filter_params(raw_params, allowed_lgb, lgb_defaults)
        point_model = lgb.LGBMRegressor(**point_params)
        cv_mae_sec, cv_used = _cv_mae_seconds_ratio(
            lambda **p: lgb.LGBMRegressor(**point_params),
            {},
            X_train, base_train, y_train_logr, y_train_sec, groups_train, desired_folds
        )
        point_model.fit(X_train, y_train_logr)
        pred_test_logr = point_model.predict(X_test) if len(X_test) else point_model.predict(X_train)

        # Quantile heads on ratio (raw, not log)
        q_models = {}
        qcfg = cfg.get("quantiles", {}) or {}
        if qcfg.get("enable", False):
            q_list = qcfg.get("q_list", [0.5, 0.9])
            for q in q_list:
                q_params = _filter_params(raw_params, allowed_lgb, lgb_defaults)
                q_params.update({"objective": "quantile", "alpha": float(q)})
                qm = lgb.LGBMRegressor(**q_params)
                qm.fit(X_train, (y_train_sec / base_train))
                q_models[str(q)] = qm

        model_type = "lightgbm"
    else:
        rf_params = _filter_params(raw_params, allowed_rf, rf_defaults)
        point_model = RandomForestRegressor(**rf_params)
        cv_mae_sec, cv_used = _cv_mae_seconds_ratio(
            lambda **p: RandomForestRegressor(**rf_params),
            {},
            X_train, base_train, y_train_logr, y_train_sec, groups_train, desired_folds
        )
        point_model.fit(X_train, y_train_logr)
        pred_test_logr = point_model.predict(X_test) if len(X_test) else point_model.predict(X_train)
        q_models = {}
        model_type = "random_forest"

    # Evaluate vs baseline
    if len(X_test) == 0:
        warnings.warn("No test samples after split; reporting train metrics as proxy.")
        y_eval_sec = y_train_sec
        base_eval = base_train
        pred_eval_logr = point_model.predict(X_train)
    else:
        y_eval_sec = y_test_sec
        base_eval = base_test
        pred_eval_logr = pred_test_logr

    pred_eval_sec = base_eval * np.exp(pred_eval_logr)
    mae = mean_absolute_error(y_eval_sec, pred_eval_sec)
    rmse = np.sqrt(mean_squared_error(y_eval_sec, pred_eval_sec))
    r2 = r2_score(y_eval_sec, pred_eval_sec)

    # baseline only
    mae_baseline = mean_absolute_error(y_eval_sec, base_eval)
    rmse_baseline = np.sqrt(mean_squared_error(y_eval_sec, base_eval))

    gain_pct = 100.0 * (mae_baseline - mae) / max(mae_baseline, 1e-9)

    # Save bundle
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type": model_type,
        "point_model": point_model,
        "quantile_models": q_models,
        "features": feature_names,
        "baseline_map": bmap,      # <<< critical for matrix prediction
        "meta": {
            "users_total": int(n_users),
            "cv_folds_requested": desired_folds,
            "cv_folds_used": int(cv_used) if cv_mae_sec else 0,
            "cv_mae_sec_mean": float(np.mean(cv_mae_sec)) if cv_mae_sec else None,
            "cv_mae_sec_std": float(np.std(cv_mae_sec)) if cv_mae_sec else None,
            "params_used": raw_params,
            "target_type": "log_time_ratio",
            "features": feature_names,
            "baseline_mae_sec": float(mae_baseline),
            "baseline_rmse_sec": float(rmse_baseline),
            "model_mae_sec": float(mae),
            "model_rmse_sec": float(rmse),
            "model_r2": float(r2),
            "model_gain_pct_over_baseline": float(gain_pct),
        },
    }
    joblib.dump(bundle, model_path)

    # Plot + report
    _barplot_metrics(fig_dir / "metrics_bar.png", mae, rmse, mae_baseline=mae_baseline)

    report = {
        "n_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_users_total": int(len(np.unique(groups))),
        "train_groups": int(len(np.unique(groups_train))),
        "cv_folds_used": bundle["meta"]["cv_folds_used"],
        "cv_mae_sec_mean": bundle["meta"]["cv_mae_sec_mean"],
        "cv_mae_sec_std": bundle["meta"]["cv_mae_sec_std"],
        "test_mae_sec": float(mae),
        "test_rmse_sec": float(rmse),
        "test_r2": float(r2),
        "baseline_mae_sec": float(mae_baseline),
        "baseline_rmse_sec": float(rmse_baseline),
        "gain_pct_over_baseline": float(gain_pct),
        "model_type": model_type,
        "model_path": str(model_path),
        "fig_metrics_bar": str((fig_dir / "metrics_bar.png").as_posix()),
        "quantiles_trained": list(q_models.keys()) if q_models else [],
        "target_type": "log_time_ratio",
        "features": feature_names,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"{model_type} saved: {model_path} | "
        f"Eval MAE={mae:.2f}s (baseline {mae_baseline:.2f}s, gain {gain_pct:+.1f}%) "
        f"RMSE={rmse:.2f}s R2={r2:.3f} | "
        f"CV MAE (sec)={report['cv_mae_sec_mean']}"
        + (f"±{report['cv_mae_sec_std']:.2f}" if report['cv_mae_sec_std'] is not None else "")
    )
    return report