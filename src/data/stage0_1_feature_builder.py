"""Build Stage 0.1 weight-based PPO features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "PPO_configurations_comparison" / "processed_final_fixed.csv"
DEFAULT_OUT = ROOT / "artifacts" / "stage0_1" / "features"
TRAIN_START = "2010-01-04"
TRAIN_END = "2021-09-30"
EPS = 1e-12


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def rolling_percentile(series: pd.Series, window: int, min_periods: int = 20) -> pd.Series:
    def _rank_last(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.nan
        return float(np.mean(values <= values[-1]))

    return series.rolling(window=window, min_periods=min_periods).apply(_rank_last, raw=True)


def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    required = {"date", "tic", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    if df.duplicated(["date", "tic"]).any():
        raise ValueError("Source has duplicate date/tic rows.")
    return df


def add_per_ticker_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "tic", "close"]].copy()
    grouped = df.groupby("tic", sort=False)

    close = df["close"].astype(float)
    log_close = np.log(np.maximum(close, EPS))
    out["logret_1d"] = grouped["close"].transform(lambda s: np.log(np.maximum(s, EPS)).diff())
    out["logret_5d"] = grouped["close"].transform(lambda s: np.log(np.maximum(s, EPS)).diff(5))
    out["logret_20d"] = grouped["close"].transform(lambda s: np.log(np.maximum(s, EPS)).diff(20))
    out["momentum_20d"] = out["logret_20d"]
    out["momentum_60d"] = grouped["close"].transform(lambda s: np.log(np.maximum(s, EPS)).diff(60))

    out["realized_vol_5d"] = grouped["close"].transform(
        lambda s: np.log(np.maximum(s, EPS)).diff().rolling(5, min_periods=2).std()
    )
    out["realized_vol_20d"] = grouped["close"].transform(
        lambda s: np.log(np.maximum(s, EPS)).diff().rolling(20, min_periods=5).std()
    )
    out["realized_vol_60d"] = grouped["close"].transform(
        lambda s: np.log(np.maximum(s, EPS)).diff().rolling(60, min_periods=10).std()
    )

    out["drawdown_20d"] = grouped["close"].transform(lambda s: s / s.rolling(20, min_periods=1).max() - 1.0)
    out["drawdown_60d"] = grouped["close"].transform(lambda s: s / s.rolling(60, min_periods=1).max() - 1.0)
    out["price_sma20_ratio"] = grouped["close"].transform(lambda s: s / s.rolling(20, min_periods=1).mean() - 1.0)
    out["price_sma60_ratio"] = grouped["close"].transform(lambda s: s / s.rolling(60, min_periods=1).mean() - 1.0)

    if {"high", "low"}.issubset(df.columns):
        out["high_low_range"] = (df["high"].astype(float) - df["low"].astype(float)) / np.maximum(close, EPS)
    if {"open"}.issubset(df.columns):
        out["close_open_return"] = close / np.maximum(df["open"].astype(float), EPS) - 1.0
        prev_close = grouped["close"].shift(1).astype(float)
        out["open_to_prev_close"] = df["open"].astype(float) / np.maximum(prev_close, EPS) - 1.0

    if "volume" in df.columns:
        log_volume = np.log1p(df["volume"].astype(float))
        out["log_volume_change_1d"] = log_volume.groupby(df["tic"]).diff()
        out["volume_zscore_20d_raw"] = log_volume.groupby(df["tic"]).transform(
            lambda s: (s - s.rolling(20, min_periods=5).mean()) / (s.rolling(20, min_periods=5).std() + EPS)
        )
        dollar_volume = np.log1p(df["volume"].astype(float) * close)
        out["dollar_volume_zscore_20d_raw"] = dollar_volume.groupby(df["tic"]).transform(
            lambda s: (s - s.rolling(20, min_periods=5).mean()) / (s.rolling(20, min_periods=5).std() + EPS)
        )

    passthrough_levels = [
        "atr_rel",
        "macd",
        "rsi_30",
        "cci_30",
        "dx_30",
        "volume_ratio",
        "obv_pct_change",
        "PE_ratio",
        "PB_ratio",
        "dividend_yield",
        "debt_ratio",
        "revenue_growth",
        "EV_EBITDA",
    ]
    for col in passthrough_levels:
        if col in df.columns:
            out[col] = df[col].astype(float)

    for col in ["macd", "rsi_30", "cci_30", "dx_30", "volume_ratio", "PE_ratio", "PB_ratio", "EV_EBITDA"]:
        if col in out.columns:
            out[f"{col}_delta_1d"] = out.groupby("tic", sort=False)[col].diff()

    return out


def add_market_features(df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    market_cols = [
        c
        for c in [
            "VIX",
            "10Y_Yield",
            "Regime_0_Prob",
            "Regime_1_Prob",
            "SP500_Trend",
            "turbulence",
            "day_sin",
            "day_cos",
        ]
        if c in df.columns
    ]
    market = df.sort_values(["date", "tic"]).groupby("date", sort=True).first()[market_cols].copy()

    if "VIX" in market:
        market["VIX_change_1d"] = market["VIX"].diff()
        market["VIX_change_5d"] = market["VIX"].diff(5)
        market["VIX_pct_change_1d"] = market["VIX"].pct_change()
        market["VIX_percentile_252"] = rolling_percentile(market["VIX"], 252)

    if "10Y_Yield" in market:
        market["yield_change_1d"] = market["10Y_Yield"].diff()
        market["yield_change_5d"] = market["10Y_Yield"].diff(5)

    if "Regime_1_Prob" in market:
        market["Regime_1_Prob_delta_1d"] = market["Regime_1_Prob"].diff()
    if {"Regime_0_Prob", "Regime_1_Prob"}.issubset(market.columns):
        probs = market[["Regime_0_Prob", "Regime_1_Prob"]].clip(EPS, 1.0)
        market["regime_entropy"] = -(probs * np.log(probs)).sum(axis=1)

    if "SP500_Trend" in market:
        market["SP500_Trend_delta_1d"] = market["SP500_Trend"].diff()
    if "turbulence" in market:
        market["turbulence_delta_1d"] = market["turbulence"].diff()
        market["turbulence_percentile_252"] = rolling_percentile(market["turbulence"], 252)

    close_panel = df.pivot(index="date", columns="tic", values="close").sort_index()
    ew_returns = close_panel.pct_change().mean(axis=1)
    market["universe_return_1d"] = ew_returns
    market["universe_return_5d"] = (1.0 + ew_returns).rolling(5, min_periods=2).apply(np.prod, raw=True) - 1.0
    market["universe_return_20d"] = (1.0 + ew_returns).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1.0
    market["universe_vol_20d"] = ew_returns.rolling(20, min_periods=5).std()

    market = market.reset_index()
    merged = out.merge(market, on="date", how="left", validate="many_to_one")
    return merged


def causal_impute(df: pd.DataFrame, feature_columns: list[str], train_mask: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out[feature_columns] = out[feature_columns].replace([np.inf, -np.inf], np.nan)
    out[feature_columns] = out.groupby("tic", sort=False)[feature_columns].ffill()
    train_medians = out.loc[train_mask, feature_columns].median(numeric_only=True)
    out[feature_columns] = out[feature_columns].fillna(train_medians).fillna(0.0)
    return out


def fit_transform_stats(df: pd.DataFrame, feature_columns: list[str], train_mask: pd.Series) -> pd.DataFrame:
    rows = []
    for col in feature_columns:
        train_values = df.loc[train_mask, col].astype(float)
        lower = float(train_values.quantile(0.01))
        upper = float(train_values.quantile(0.99))
        clipped = train_values.clip(lower, upper)
        mean = float(clipped.mean())
        std = float(clipped.std(ddof=0))
        if not np.isfinite(std) or std < EPS:
            std = 1.0
        rows.append(
            {
                "feature": col,
                "lower_quantile": 0.01,
                "upper_quantile": 0.99,
                "lower_value_train": lower,
                "upper_value_train": upper,
                "mean_after_clip_train": mean,
                "std_after_clip_train": std,
            }
        )
    return pd.DataFrame(rows)


def apply_transform(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for row in stats.itertuples(index=False):
        clipped = out[row.feature].astype(float).clip(row.lower_value_train, row.upper_value_train)
        out[row.feature] = (clipped - row.mean_after_clip_train) / row.std_after_clip_train
    return out


def diagnostics(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "feature_count": int(len(feature_columns)),
                "ticker_count": int(df["tic"].nunique()),
                "date_count": int(df["date"].nunique()),
                "start_date": str(pd.to_datetime(df["date"]).min().date()),
                "end_date": str(pd.to_datetime(df["date"]).max().date()),
                "missing_feature_cells": int(df[feature_columns].isna().sum().sum()),
                "inf_feature_cells": int(np.isinf(df[feature_columns].to_numpy(dtype=float)).sum()),
                "duplicate_date_tic_rows": int(df.duplicated(["date", "tic"]).sum()),
            }
        ]
    )


def build_features(source: Path, out_dir: Path, train_start: str, train_end: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_source(source)
    engineered = add_per_ticker_features(df)
    engineered = add_market_features(df, engineered)
    engineered = engineered.sort_values(["date", "tic"]).reset_index(drop=True)

    feature_columns = [c for c in engineered.columns if c not in {"date", "tic", "close"}]
    train_mask = (engineered["date"] >= pd.Timestamp(train_start)) & (engineered["date"] <= pd.Timestamp(train_end))
    engineered = causal_impute(engineered, feature_columns, train_mask)
    stats = fit_transform_stats(engineered, feature_columns, train_mask)
    model_ready = apply_transform(engineered, stats)

    raw_path = out_dir / "stage0_1_weight_features_raw.csv"
    model_ready_path = out_dir / "stage0_1_weight_features_model_ready.csv"
    stats_path = out_dir / "stage0_1_weight_feature_transform_stats.csv"
    raw_diag_path = out_dir / "stage0_1_weight_features_raw_diagnostics.csv"
    ready_diag_path = out_dir / "stage0_1_weight_features_model_ready_diagnostics.csv"
    manifest_path = out_dir / "stage0_1_weight_feature_manifest.json"

    engineered.to_csv(raw_path, index=False)
    model_ready.to_csv(model_ready_path, index=False)
    stats.to_csv(stats_path, index=False)
    diagnostics(engineered, feature_columns).to_csv(raw_diag_path, index=False)
    diagnostics(model_ready, feature_columns).to_csv(ready_diag_path, index=False)

    manifest = {
        "source_csv": rel(source),
        "raw_csv": rel(raw_path),
        "model_ready_csv": rel(model_ready_path),
        "transform_stats_csv": rel(stats_path),
        "raw_diagnostics_csv": rel(raw_diag_path),
        "model_ready_diagnostics_csv": rel(ready_diag_path),
        "train_start": train_start,
        "train_end": train_end,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "transform": "train-window 1%-99% winsorization followed by train-window z-score",
        "causality_notes": [
            "Rolling features use current and past daily observations only.",
            "Initial rolling/change NaNs are filled by causal ticker forward-fill and train-window medians.",
            "The model-ready CSV uses the final Stage 0 train window for normalization; for strict walk-forward validation, fit scalers per fold before final overnight comparison.",
            "GRU forecast columns are intentionally excluded from this first Stage 0.1 interpretable feature set.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end", default=TRAIN_END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_features(
        source=Path(args.source),
        out_dir=Path(args.out_dir),
        train_start=args.train_start,
        train_end=args.train_end,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
