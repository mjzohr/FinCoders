from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import PanelBundle, SequenceStandardizer


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_feature_metadata(config: ExperimentConfig, bundle: PanelBundle) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "task": config.task,
        "modalities": config.modalities,
        "horizon": config.horizon,
        "lookback": config.lookback,
        "news_lag_days": config.news_lag_days,
        "eval_mode": config.eval_mode,
        "seed": config.seed,
        "price_feature_cols": list(bundle.price_feature_cols),
        "news_feature_cols": list(bundle.news_feature_cols),
        "use_news": config.modalities == "price_news",
    }


def save_standardizer(path: Path, standardizer: SequenceStandardizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        price_mean=standardizer.price_mean,
        price_std=standardizer.price_std,
        news_mean=standardizer.news_mean,
        news_std=standardizer.news_std,
    )


def load_standardizer(path: Path) -> SequenceStandardizer:
    arrays = np.load(path)
    return SequenceStandardizer(
        price_mean=arrays["price_mean"],
        price_std=arrays["price_std"],
        news_mean=arrays["news_mean"],
        news_std=arrays["news_std"],
    )


def default_metric_for_task(task: str) -> tuple[str, bool]:
    if task == "classification":
        return "test_balanced_accuracy", True
    return "test_daily_spearman_ic_mean", True


def select_best_run(
    artifacts_dir: Path,
    task: str | None = None,
    metric: str | None = None,
    maximize: bool | None = None,
) -> tuple[Path, dict[str, object], float]:
    run_dirs = [path.parent for path in artifacts_dir.rglob("summary.csv") if (path.parent / "config.json").exists()]
    if not run_dirs:
        raise FileNotFoundError(f"No completed runs found under {artifacts_dir}")

    rows: list[dict[str, object]] = []
    for run_dir in sorted(set(run_dirs)):
        config = ExperimentConfig.from_dict(load_json(run_dir / "config.json"))
        if task is not None and config.task != task:
            continue
        summary = pd.read_csv(run_dir / "summary.csv")
        if summary.empty:
            continue
        chosen_metric, chosen_maximize = default_metric_for_task(config.task)
        chosen_metric = metric or chosen_metric
        chosen_maximize = chosen_maximize if maximize is None else maximize
        if chosen_metric not in summary.columns:
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "config": config,
                "metric": chosen_metric,
                "maximize": chosen_maximize,
                "score": float(summary[chosen_metric].mean()),
            }
        )

    if not rows:
        raise ValueError("No runs matched the requested task/metric filter.")

    rows.sort(key=lambda row: row["score"], reverse=bool(rows[0]["maximize"]))
    best = rows[0]
    return Path(best["run_dir"]), best["config"], float(best["score"])
