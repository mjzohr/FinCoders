from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .artifacts import load_json, load_standardizer
from .config import ExperimentConfig
from .data import build_model_panel, build_tabular_matrix, create_latest_samples
from .models import TorchSequenceDataset
from .train import build_neural_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fresh daily data using a trained champion artifact.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Path to a champion directory.")
    parser.add_argument("--price-csv", type=Path, default=None)
    parser.add_argument("--news-csv", type=Path, default=None)
    parser.add_argument("--as-of-date", type=str, default=None, help="Optional cutoff date, e.g. 2024-11-20.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def parse_optional_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse as-of date: {value}")
    return pd.Timestamp(parsed)


def load_config(artifact_dir: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(load_json(artifact_dir / "config.json"))


def load_lightgbm_model(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def score_regression_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["predicted_return"] = frame["prediction"]
    frame["predicted_direction"] = np.where(frame["prediction"] >= 0.0, "up", "down")
    frame["score_rank"] = frame["prediction"].rank(method="first", ascending=False).astype(int)
    frame["score_percentile"] = frame["prediction"].rank(pct=True, ascending=True)
    return frame.sort_values("prediction", ascending=False).reset_index(drop=True)


def score_classification_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["up_probability"] = frame["prediction"]
    frame["predicted_label"] = (frame["prediction"] >= 0.5).astype(int)
    frame["score_rank"] = frame["prediction"].rank(method="first", ascending=False).astype(int)
    frame["score_percentile"] = frame["prediction"].rank(pct=True, ascending=True)
    return frame.sort_values("prediction", ascending=False).reset_index(drop=True)


def predict_neural_live(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            price_seq = batch["price_seq"].to(device)
            news_seq = batch["news_seq"].to(device)
            logits = model(price_seq, news_seq)
            batch_prediction = torch.sigmoid(logits) if task == "classification" else logits
            predictions.append(batch_prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    config = load_config(artifact_dir)
    if args.price_csv is not None:
        config.price_csv = args.price_csv
    if args.news_csv is not None:
        config.news_csv = args.news_csv
    if args.device is not None:
        config.device = args.device

    as_of_date = parse_optional_date(args.as_of_date)
    panel_bundle = build_model_panel(
        price_csv=config.price_csv,
        news_csv=config.news_csv if config.modalities == "price_news" else None,
        news_lag_days=config.news_lag_days,
        horizons=(config.horizon,),
    )
    live_samples = create_latest_samples(
        bundle=panel_bundle,
        lookback=config.lookback,
        use_news=config.modalities == "price_news",
        as_of_date=as_of_date,
    )
    if live_samples.size == 0:
        raise ValueError("No live samples were created. Check lookback size and input files.")

    if config.model_name == "lightgbm":
        model = load_lightgbm_model(artifact_dir / "model.txt")
        features = build_tabular_matrix(live_samples, include_news=config.modalities == "price_news")
        predictions = np.asarray(model.predict(features), dtype=np.float32)
    else:
        standardizer = load_standardizer(artifact_dir / "standardizer.npz")
        scaled_samples = standardizer.transform(live_samples)
        device = resolve_device(config.device)
        model = build_neural_model(
            config=config,
            price_dim=scaled_samples.price_seq.shape[-1],
            news_dim=scaled_samples.news_seq.shape[-1],
        ).to(device)
        state_dict = torch.load(artifact_dir / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        loader = DataLoader(
            TorchSequenceDataset(scaled_samples.price_seq, scaled_samples.news_seq, scaled_samples.targets),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )
        predictions = predict_neural_live(model, loader, device, config.task)

    prediction_frame = pd.DataFrame(
        {
            "sequence_end_date": pd.to_datetime(live_samples.dates),
            "ticker": live_samples.tickers,
            "prediction": predictions,
            "model_name": config.model_name,
            "task": config.task,
            "horizon": config.horizon,
            "lookback": config.lookback,
            "modalities": config.modalities,
            "news_lag_days": config.news_lag_days,
        }
    )
    if config.task == "classification":
        prediction_frame = score_classification_predictions(prediction_frame)
    else:
        prediction_frame = score_regression_predictions(prediction_frame)

    scoring_date = str(prediction_frame["sequence_end_date"].max().date())
    output_path = args.output_path or artifact_dir / f"live_predictions_{scoring_date}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)

    preview = prediction_frame.head(args.top_k)
    print(preview.to_string(index=False))
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "output_path": str(output_path),
                "scoring_date": scoring_date,
                "rows_scored": int(len(prediction_frame)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
