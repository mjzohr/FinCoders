from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import build_feature_metadata, default_metric_for_task, save_json, save_standardizer, select_best_run
from .config import ExperimentConfig
from .data import build_model_panel, create_sequence_samples, fit_standardizer
from .evaluation import compute_metrics
from .models import TorchSequenceDataset
from .train import (
    predict_lightgbm,
    predict_neural,
    resolve_device,
    run_experiment,
    seed_everything,
    train_lightgbm,
    train_neural_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain a selected experiment config into one deployable champion artifact.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--champions-dir", type=Path, default=Path("champions"))
    parser.add_argument("--source-run", type=Path, default=None, help="Path to an existing experiment directory under artifacts.")
    parser.add_argument("--run-name", type=str, default=None, help="Name of an existing experiment directory under artifacts.")
    parser.add_argument("--task", choices=["regression", "classification"], default=None)
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--maximize", dest="maximize", action="store_true")
    parser.add_argument("--minimize", dest="maximize", action="store_false")
    parser.set_defaults(maximize=None)
    parser.add_argument("--name", type=str, default=None, help="Optional name for the champion artifact directory.")
    parser.add_argument("--price-csv", type=Path, default=None)
    parser.add_argument("--news-csv", type=Path, default=None)
    parser.add_argument("--val-days", type=int, default=None, help="Override validation window used for final early stopping.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_source_config(args: argparse.Namespace) -> tuple[Path, ExperimentConfig, str, float]:
    if args.source_run is not None:
        run_dir = args.source_run
        config = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        metric, maximize = default_metric_for_task(config.task)
        metric = args.metric or metric
        summary = pd.read_csv(run_dir / "summary.csv")
        score = float(summary[metric].mean()) if metric in summary.columns else float("nan")
        return run_dir, config, metric, score

    if args.run_name is not None:
        run_dir = args.artifacts_dir / args.run_name
        config = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        metric, maximize = default_metric_for_task(config.task)
        metric = args.metric or metric
        summary = pd.read_csv(run_dir / "summary.csv")
        score = float(summary[metric].mean()) if metric in summary.columns else float("nan")
        return run_dir, config, metric, score

    desired_task = args.task or "regression"
    metric = args.metric or default_metric_for_task(desired_task)[0]
    run_dir, config, score = select_best_run(
        artifacts_dir=args.artifacts_dir,
        task=args.task,
        metric=metric,
        maximize=args.maximize,
    )
    return run_dir, config, metric, score


def make_recent_train_val_masks(sample_dates: np.ndarray, val_days: int) -> tuple[np.ndarray, np.ndarray]:
    unique_dates = np.sort(np.unique(sample_dates))
    if len(unique_dates) <= val_days:
        raise ValueError("Not enough sample dates to carve out a champion validation window.")
    train_dates = unique_dates[:-val_days]
    val_dates = unique_dates[-val_days:]
    return np.isin(sample_dates, train_dates), np.isin(sample_dates, val_dates)


def champion_name(source_run_name: str, config: ExperimentConfig, override: str | None) -> str:
    if override:
        return override
    return f"{source_run_name}__champion"


def main() -> None:
    args = parse_args()
    source_run_dir, config, source_metric, source_score = load_source_config(args)

    if args.price_csv is not None:
        config.price_csv = args.price_csv
    if args.news_csv is not None:
        config.news_csv = args.news_csv
    if args.device is not None:
        config.device = args.device

    val_days = args.val_days or config.val_days
    seed_everything(config.seed)

    panel_bundle = build_model_panel(
        price_csv=config.price_csv,
        news_csv=config.news_csv if config.modalities == "price_news" else None,
        news_lag_days=config.news_lag_days,
        horizons=(config.horizon,),
    )
    samples = create_sequence_samples(
        bundle=panel_bundle,
        horizon=config.horizon,
        lookback=config.lookback,
        task=config.task,
        flat_threshold=config.flat_threshold,
        use_news=config.modalities == "price_news",
        target_clip=config.target_clip if config.task == "regression" else None,
    )
    if samples.size == 0:
        raise ValueError("No samples were created for champion training.")

    train_mask, val_mask = make_recent_train_val_masks(samples.dates, val_days=val_days)
    train_samples = samples.subset(train_mask)
    val_samples = samples.subset(val_mask)

    champion_dir = args.champions_dir / champion_name(source_run_dir.name, config, args.name)
    champion_dir.mkdir(parents=True, exist_ok=True)

    save_json(champion_dir / "config.json", config.as_dict())
    save_json(champion_dir / "feature_metadata.json", build_feature_metadata(config, panel_bundle))
    save_json(
        champion_dir / "selection.json",
        {
            "source_run": str(source_run_dir),
            "source_run_name": source_run_dir.name,
            "source_metric": source_metric,
            "source_metric_score": source_score,
            "validation_days": val_days,
            "champion_dir": str(champion_dir),
        },
    )

    if config.model_name == "lightgbm":
        model, val_metrics = train_lightgbm(config, train_samples, val_samples)
        val_predictions = predict_lightgbm(model, config, val_samples)
        model.save_model(str(champion_dir / "model.txt"))
    else:
        standardizer = fit_standardizer(train_samples)
        scaled_train = standardizer.transform(train_samples)
        scaled_val = standardizer.transform(val_samples)
        model, val_metrics = train_neural_model(config, scaled_train, scaled_val)
        criterion = nn.BCEWithLogitsLoss() if config.task == "classification" else nn.HuberLoss(delta=0.02)
        val_loader = DataLoader(
            TorchSequenceDataset(scaled_val.price_seq, scaled_val.news_seq, scaled_val.targets),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        val_predictions, _ = predict_neural(model, val_loader, criterion, resolve_device(config.device), config.task)
        torch.save(model.state_dict(), champion_dir / "model.pt")
        save_standardizer(champion_dir / "standardizer.npz", standardizer)

    validation_metrics = compute_metrics(config.task, val_samples.targets, val_predictions, val_samples.dates)
    validation_metrics.update(val_metrics)

    validation_predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(val_samples.dates),
            "ticker": val_samples.tickers,
            "target": val_samples.targets,
            "prediction": val_predictions,
        }
    )
    validation_predictions.to_csv(champion_dir / "validation_predictions.csv", index=False)

    save_json(
        champion_dir / "champion_summary.json",
        {
            "champion_name": champion_dir.name,
            "model_name": config.model_name,
            "task": config.task,
            "modalities": config.modalities,
            "horizon": config.horizon,
            "lookback": config.lookback,
            "news_lag_days": config.news_lag_days,
            "validation_days": val_days,
            "train_sample_count": train_samples.size,
            "validation_sample_count": val_samples.size,
            "validation_metrics": validation_metrics,
            "latest_validation_date": str(pd.to_datetime(val_samples.dates).max().date()),
        },
    )

    print(
        json.dumps(
            {
                "champion_dir": str(champion_dir),
                "source_run": str(source_run_dir),
                "source_metric": source_metric,
                "source_metric_score": source_score,
                "validation_metrics": validation_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
