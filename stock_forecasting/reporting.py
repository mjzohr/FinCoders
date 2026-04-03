from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REGRESSION_RULES = [
    ("test_daily_spearman_ic_mean", 1.60, 1),
    ("test_spearman_ic", 1.10, 1),
    ("test_directional_accuracy", 0.90, 1),
    ("test_top_bottom_decile_spread", 1.00, 1),
    ("test_rmse", 0.85, -1),
    ("test_mae", 0.60, -1),
]

CLASSIFICATION_RULES = [
    ("test_balanced_accuracy", 1.60, 1),
    ("test_f1", 1.20, 1),
    ("test_accuracy", 0.90, 1),
    ("test_precision", 0.60, 1),
    ("test_recall", 0.60, 1),
]


def horizon_label(horizon: int) -> str:
    return {1: "Daily", 5: "Weekly", 21: "Monthly"}.get(horizon, f"Horizon {horizon}")


def discover_run_dirs(results_root: Path = Path("artifacts"), search_root: Path = Path(".")) -> list[Path]:
    if results_root.exists():
        summary_paths = list(results_root.rglob("summary.csv"))
    else:
        summary_paths = []

    if not summary_paths:
        summary_paths = [
            path for path in search_root.rglob("summary.csv") if "artifacts" in path.parts
        ]

    run_dirs = []
    for summary_path in summary_paths:
        run_dir = summary_path.parent
        if (run_dir / "config.json").exists():
            run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main_metric_for_task(task: str, frame: pd.DataFrame) -> str | None:
    candidates = [
        "test_daily_spearman_ic_mean",
        "test_balanced_accuracy",
        "test_spearman_ic",
        "test_f1",
        "test_directional_accuracy",
        "test_accuracy",
    ]
    preferred = (
        ["test_balanced_accuracy", "test_f1", "test_accuracy"]
        if task == "classification"
        else ["test_daily_spearman_ic_mean", "test_spearman_ic", "test_directional_accuracy"]
    )
    for column in preferred + candidates:
        if column in frame.columns:
            return column
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(numeric)), index=numeric.index)
    return (numeric - numeric.mean()) / std


def load_results(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []

    for run_dir in run_dirs:
        config = load_json(run_dir / "config.json")
        fold_df = pd.read_csv(run_dir / "summary.csv")
        numeric_cols = fold_df.select_dtypes(include="number").columns.tolist()

        run_row: dict[str, object] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": config.get("model_name"),
            "modalities": config.get("modalities"),
            "task": config.get("task"),
            "horizon": config.get("horizon"),
            "lookback": config.get("lookback"),
            "news_lag_days": config.get("news_lag_days"),
            "seed": config.get("seed"),
            "eval_mode": config.get("eval_mode"),
            "price_csv": config.get("price_csv"),
            "news_csv": config.get("news_csv"),
            "fold_count": len(fold_df),
        }
        for column in numeric_cols:
            run_row[column] = fold_df[column].mean()
            run_row[f"{column}__std"] = fold_df[column].std(ddof=0)
        run_rows.append(run_row)

        if not fold_df.empty:
            fold_frames.append(
                fold_df.assign(
                    run_name=run_dir.name,
                    run_dir=str(run_dir),
                    model_name=config.get("model_name"),
                    modalities=config.get("modalities"),
                    task=config.get("task"),
                    horizon=config.get("horizon"),
                    lookback=config.get("lookback"),
                    news_lag_days=config.get("news_lag_days"),
                    seed=config.get("seed"),
                    eval_mode=config.get("eval_mode"),
                    price_csv=config.get("price_csv"),
                    news_csv=config.get("news_csv"),
                )
            )

    runs_df = pd.DataFrame(run_rows)
    folds_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    return runs_df, folds_df


def rank_runs(runs_df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if runs_df.empty:
        return runs_df.copy(), None

    task = (
        runs_df["task"].mode().iloc[0]
        if "task" in runs_df.columns and not runs_df["task"].dropna().empty
        else "regression"
    )
    rules = CLASSIFICATION_RULES if task == "classification" else REGRESSION_RULES
    main_metric = main_metric_for_task(task, runs_df)

    ranked = runs_df.copy()
    composite = pd.Series(np.zeros(len(ranked)), index=ranked.index, dtype=float)
    used_columns = []
    for column, weight, direction in rules:
        if column in ranked.columns:
            column_score = zscore(ranked[column]).fillna(0.0)
            if direction < 0:
                column_score = -column_score
            composite = composite + weight * column_score
            used_columns.append(column)

    if main_metric and f"{main_metric}__std" in ranked.columns:
        composite = composite - 0.35 * zscore(ranked[f"{main_metric}__std"]).fillna(0.0)

    ranked["composite_score"] = composite
    ranked["main_metric"] = main_metric
    ranked["used_metric_count"] = len(used_columns)
    ranked = ranked.sort_values(["composite_score", main_metric], ascending=[False, False], na_position="last")
    return ranked.reset_index(drop=True), main_metric


def load_predictions_for_run(run_name: str, runs_df: pd.DataFrame) -> pd.DataFrame:
    row = runs_df.loc[runs_df["run_name"] == run_name]
    if row.empty:
        return pd.DataFrame()
    run_dir = Path(row.iloc[0]["run_dir"])
    frames = []
    for prediction_path in sorted(run_dir.glob("*/test_predictions.csv")):
        frame = pd.read_csv(prediction_path, parse_dates=["date"])
        frame["fold"] = prediction_path.parent.name
        frame["run_name"] = run_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_prediction_diagnostics(
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    frame = predictions_df.copy()
    frame["abs_error"] = (frame["prediction"] - frame["target"]).abs()
    frame["direction_hit"] = ((frame["prediction"] > 0) == (frame["target"] > 0)).astype(float)

    daily_rows = []
    for date, group in frame.groupby("date"):
        if len(group) < 5:
            continue
        bucket = max(1, len(group) // 10)
        ordered = group.sort_values("prediction")
        try:
            daily_ic = group["prediction"].corr(group["target"], method="spearman")
        except Exception:
            daily_ic = np.nan
        daily_rows.append(
            {
                "date": date,
                "daily_ic": daily_ic,
                "top_bottom_spread": ordered["target"].tail(bucket).mean()
                - ordered["target"].head(bucket).mean(),
                "directional_accuracy": group["direction_hit"].mean(),
                "mean_abs_error": group["abs_error"].mean(),
                "sample_count": len(group),
            }
        )

    daily_df = pd.DataFrame(daily_rows).sort_values("date") if daily_rows else pd.DataFrame()
    if not daily_df.empty:
        daily_df["cumulative_spread"] = daily_df["top_bottom_spread"].fillna(0.0).cumsum()
        daily_df["rolling_ic_20"] = daily_df["daily_ic"].rolling(20, min_periods=5).mean()

    ticker_df = (
        frame.groupby("ticker")
        .agg(
            observations=("ticker", "size"),
            mean_abs_error=("abs_error", "mean"),
            directional_accuracy=("direction_hit", "mean"),
            prediction_mean=("prediction", "mean"),
            target_mean=("target", "mean"),
        )
        .reset_index()
        .sort_values(["directional_accuracy", "mean_abs_error"], ascending=[False, True])
    )

    magnitude_df = pd.DataFrame()
    if frame["target"].notna().sum() > 10:
        tmp = frame.copy()
        tmp["target_magnitude"] = tmp["target"].abs()
        bucket_count = min(5, tmp["target_magnitude"].nunique())
        if bucket_count >= 2:
            tmp["magnitude_bucket"] = pd.qcut(tmp["target_magnitude"], q=bucket_count, duplicates="drop")
            magnitude_df = (
                tmp.groupby("magnitude_bucket")
                .agg(
                    mean_abs_error=("abs_error", "mean"),
                    directional_accuracy=("direction_hit", "mean"),
                    avg_target_magnitude=("target_magnitude", "mean"),
                    observations=("target_magnitude", "size"),
                )
                .reset_index()
            )

    return daily_df, ticker_df, magnitude_df


def summarize_horizon(
    ranked_df: pd.DataFrame,
    horizon: int,
    main_metric: str,
) -> dict[str, object] | None:
    horizon_df = ranked_df.loc[ranked_df["horizon"] == horizon].copy()
    if horizon_df.empty:
        return None

    best = horizon_df.iloc[0]
    runner_up = horizon_df.iloc[1] if len(horizon_df) > 1 else None
    summary: dict[str, object] = {
        "label": horizon_label(horizon),
        "horizon": horizon,
        "best": best,
        "runner_up": runner_up,
        "main_metric": main_metric,
        "main_metric_value": float(best[main_metric]) if main_metric in best and pd.notna(best[main_metric]) else np.nan,
        "composite_score": float(best["composite_score"]) if pd.notna(best["composite_score"]) else np.nan,
        "stability": float(best.get(f"{main_metric}__std", np.nan)),
        "directional_accuracy": float(best.get("test_directional_accuracy", np.nan)),
        "spread": float(best.get("test_top_bottom_decile_spread", np.nan)),
    }
    if runner_up is not None and main_metric in horizon_df.columns:
        summary["metric_gap_to_runner_up"] = float(best.get(main_metric, np.nan) - runner_up.get(main_metric, np.nan))
    else:
        summary["metric_gap_to_runner_up"] = np.nan
    return summary
