from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import build_feature_metadata, save_json, save_standardizer
from .config import DEFAULT_HORIZONS, ExperimentConfig
from .data import (
    InsufficientDataError,
    PanelBundle,
    SequenceSamples,
    build_model_panel,
    build_tabular_matrix,
    create_sequence_samples,
    fit_standardizer,
    make_splits,
)
from .evaluation import compute_metrics
from .models import HuggingFacePatchTSTModel, LSTMFusionModel, TimeXerFusionModel, TorchSequenceDataset


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train stock forecasting models on price and news data.")
    parser.add_argument("--price-csv", type=Path, default=Path("data/dates_on_left_stock_data.csv"))
    parser.add_argument("--news-csv", type=Path, default=Path("data/news_all_sentiment.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--model",
        dest="model_name",
        choices=["lightgbm", "lstm", "timexer", "hf_patchtst"],
        default="lightgbm",
    )
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--modalities", choices=["price", "price_news"], default="price_news")
    parser.add_argument("--horizon", type=int, choices=DEFAULT_HORIZONS, default=1)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--news-lag-days", type=int, default=1)
    parser.add_argument("--eval-mode", choices=["holdout", "walkforward"], default="walkforward")
    parser.add_argument("--min-train-days", type=int, default=126)
    parser.add_argument("--val-days", type=int, default=21)
    parser.add_argument("--test-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--patch-len", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--flat-threshold", type=float, default=0.0)
    parser.add_argument("--target-clip", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    namespace = parser.parse_args()
    return ExperimentConfig(**vars(namespace))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lightgbm(
    config: ExperimentConfig,
    train_samples: SequenceSamples,
    val_samples: SequenceSamples,
) -> tuple[object, dict[str, float]]:
    import lightgbm as lgb

    include_news = config.modalities == "price_news"
    x_train = build_tabular_matrix(train_samples, include_news=include_news)
    x_val = build_tabular_matrix(val_samples, include_news=include_news)

    params = {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_data_in_leaf": 32,
        "verbosity": -1,
        "seed": config.seed,
    }

    if config.task == "classification":
        positive = float(train_samples.targets.sum())
        negative = float(len(train_samples.targets) - positive)
        params.update(
            {
                "objective": "binary",
                "metric": "binary_logloss",
                "scale_pos_weight": negative / max(positive, 1.0),
            }
        )
    else:
        params.update({"objective": "regression", "metric": "l2"})

    train_dataset = lgb.Dataset(x_train, label=train_samples.targets)
    valid_dataset = lgb.Dataset(x_val, label=val_samples.targets, reference=train_dataset)

    booster = lgb.train(
        params=params,
        train_set=train_dataset,
        valid_sets=[valid_dataset],
        valid_names=["val"],
        num_boost_round=config.num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds),
            lgb.log_evaluation(period=25),
        ],
    )

    val_predictions = booster.predict(x_val, num_iteration=booster.best_iteration)
    metrics = compute_metrics(config.task, val_samples.targets, np.asarray(val_predictions), val_samples.dates)
    return booster, metrics


def predict_lightgbm(
    booster: object,
    config: ExperimentConfig,
    samples: SequenceSamples,
) -> np.ndarray:
    include_news = config.modalities == "price_news"
    x = build_tabular_matrix(samples, include_news=include_news)
    return np.asarray(booster.predict(x, num_iteration=booster.best_iteration), dtype=np.float32)


def make_loaders(
    config: ExperimentConfig,
    train_samples: SequenceSamples,
    val_samples: SequenceSamples,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TorchSequenceDataset(train_samples.price_seq, train_samples.news_seq, train_samples.targets)
    val_dataset = TorchSequenceDataset(val_samples.price_seq, val_samples.news_seq, val_samples.targets)
    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=False,
        ),
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        ),
    )


def build_neural_model(config: ExperimentConfig, price_dim: int, news_dim: int) -> nn.Module:
    use_news = config.modalities == "price_news"
    if config.model_name == "lstm":
        return LSTMFusionModel(
            price_dim=price_dim,
            news_dim=news_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            use_news=use_news,
        )
    if config.model_name == "timexer":
        return TimeXerFusionModel(
            price_dim=price_dim,
            news_dim=news_dim,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.transformer_layers,
            patch_len=config.patch_len,
            dropout=config.dropout,
            use_news=use_news,
        )
    if config.model_name == "hf_patchtst":
        return HuggingFacePatchTSTModel(
            price_dim=price_dim,
            news_dim=news_dim,
            context_length=config.lookback,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.transformer_layers,
            patch_len=config.patch_len,
            dropout=config.dropout,
            use_news=use_news,
            task=config.task,
        )
    raise ValueError(f"Unsupported neural model: {config.model_name}")


def train_neural_model(
    config: ExperimentConfig,
    train_samples: SequenceSamples,
    val_samples: SequenceSamples,
) -> tuple[nn.Module, dict[str, float]]:
    device = resolve_device(config.device)
    model = build_neural_model(
        config=config,
        price_dim=train_samples.price_seq.shape[-1],
        news_dim=train_samples.news_seq.shape[-1],
    ).to(device)

    train_loader, val_loader = make_loaders(config, train_samples, val_samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.task == "classification":
        positive = float(train_samples.targets.sum())
        negative = float(len(train_samples.targets) - positive)
        pos_weight = torch.tensor([negative / max(positive, 1.0)], device=device)
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.HuberLoss(delta=0.02)

    best_state = None
    best_loss = float("inf")
    best_metrics: dict[str, float] = {}
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            price_seq = batch["price_seq"].to(device)
            news_seq = batch["news_seq"].to(device)
            target = batch["target"].to(device)
            prediction = model(price_seq, news_seq)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = int(target.shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        val_predictions, val_loss = predict_neural(model, val_loader, criterion, device, config.task)
        metrics = compute_metrics(config.task, val_samples.targets, val_predictions, val_samples.dates)
        metrics["validation_loss"] = float(val_loss)

        train_loss = running_loss / max(sample_count, 1)
        main_metric = _main_metric_name(config.task)
        print(
            f"epoch={epoch + 1:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} {main_metric}={metrics.get(main_metric, 0.0):.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_metrics = metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Neural model failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return model, best_metrics


def predict_neural(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task: str,
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            price_seq = batch["price_seq"].to(device)
            news_seq = batch["news_seq"].to(device)
            target = batch["target"].to(device)
            logits = model(price_seq, news_seq)
            loss = criterion(logits, target)
            losses.append(float(loss.item()) * int(target.shape[0]))
            batch_prediction = torch.sigmoid(logits) if task == "classification" else logits
            predictions.append(batch_prediction.cpu().numpy())

    stacked = np.concatenate(predictions, axis=0).astype(np.float32)
    average_loss = sum(losses) / max(len(loader.dataset), 1)
    return stacked, average_loss


def save_predictions(
    path: Path,
    samples: SequenceSamples,
    predictions: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(samples.dates),
            "ticker": samples.tickers,
            "target": samples.targets,
            "prediction": predictions,
        }
    )
    frame.to_csv(path, index=False)


def run_experiment(config: ExperimentConfig, panel_bundle: PanelBundle | None = None) -> dict[str, object]:
    seed_everything(config.seed)
    panel_bundle = panel_bundle or build_model_panel(
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
        raise ValueError("No samples were created. Try a shorter lookback or verify the input CSVs.")

    splits = make_splits(
        sample_dates=samples.dates,
        eval_mode=config.eval_mode,
        min_train_days=config.min_train_days,
        val_days=config.val_days,
        test_days=config.test_days,
        step_days=config.step_days,
    )

    experiment_dir = config.output_dir / config.run_name()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_json(experiment_dir / "config.json", config.as_dict())
    save_json(experiment_dir / "feature_metadata.json", build_feature_metadata(config, panel_bundle))

    fold_rows: list[dict[str, object]] = []

    for split in splits:
        fold_dir = experiment_dir / split.name
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_samples = samples.subset(split.train_mask)
        val_samples = samples.subset(split.val_mask)
        test_samples = samples.subset(split.test_mask)
        if min(train_samples.size, val_samples.size, test_samples.size) == 0:
            raise ValueError(f"Split {split.name} produced an empty train/val/test partition.")

        if config.model_name == "lightgbm":
            model, val_metrics = train_lightgbm(config, train_samples, val_samples)
            test_predictions = predict_lightgbm(model, config, test_samples)
            model.save_model(str(fold_dir / "model.txt"))
        else:
            standardizer = fit_standardizer(train_samples)
            scaled_train = standardizer.transform(train_samples)
            scaled_val = standardizer.transform(val_samples)
            scaled_test = standardizer.transform(test_samples)

            model, val_metrics = train_neural_model(config, scaled_train, scaled_val)
            criterion = nn.BCEWithLogitsLoss() if config.task == "classification" else nn.HuberLoss(delta=0.02)
            test_loader = DataLoader(
                TorchSequenceDataset(scaled_test.price_seq, scaled_test.news_seq, scaled_test.targets),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
            test_predictions, _ = predict_neural(model, test_loader, criterion, resolve_device(config.device), config.task)
            torch.save(model.state_dict(), fold_dir / "model.pt")
            save_standardizer(fold_dir / "standardizer.npz", standardizer)

        test_metrics = compute_metrics(config.task, test_samples.targets, test_predictions, test_samples.dates)
        save_predictions(fold_dir / "test_predictions.csv", test_samples, test_predictions)

        fold_row: dict[str, object] = {"fold": split.name}
        for key, value in val_metrics.items():
            fold_row[f"val_{key}"] = value
        for key, value in test_metrics.items():
            fold_row[f"test_{key}"] = value
        fold_rows.append(fold_row)

    summary_frame = pd.DataFrame(fold_rows)
    summary_frame.to_csv(experiment_dir / "summary.csv", index=False)

    summary = {
        "run_name": config.run_name(),
        "fold_count": len(fold_rows),
        "average_metrics": summary_frame.mean(numeric_only=True).to_dict(),
        "output_dir": str(experiment_dir),
    }
    save_json(experiment_dir / "summary.json", summary)
    return summary


def _main_metric_name(task: str) -> str:
    return "balanced_accuracy" if task == "classification" else "daily_spearman_ic_mean"


def main() -> None:
    config = parse_args()
    try:
        summary = run_experiment(config)
    except InsufficientDataError as exc:
        summary = {
            "status": "skipped",
            "run_name": config.run_name(),
            "reason": str(exc),
        }
        print(json.dumps(summary, indent=2))
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
