from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import DEFAULT_HORIZONS, ExperimentConfig
from .data import build_model_panel
from .train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model x horizon experiment grid.")
    parser.add_argument("--price-csv", type=Path, default=Path("data/dates_on_left_stock_data.csv"))
    parser.add_argument("--news-csv", type=Path, default=Path("data/news_all_sentiment.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lightgbm", "lstm", "timexer", "hf_patchtst"],
        default=["lightgbm", "lstm", "timexer", "hf_patchtst"],
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--modalities", nargs="+", choices=["price", "price_news"], default=["price", "price_news"])
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--eval-mode", choices=["holdout", "walkforward"], default="walkforward")
    parser.add_argument("--news-lag-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for modalities in args.modalities:
        panel_bundle = build_model_panel(
            price_csv=args.price_csv,
            news_csv=args.news_csv if modalities == "price_news" else None,
            news_lag_days=args.news_lag_days,
            horizons=tuple(args.horizons),
        )
        for horizon in args.horizons:
            for model_name in args.models:
                config = ExperimentConfig(
                    price_csv=args.price_csv,
                    news_csv=args.news_csv,
                    output_dir=output_dir,
                    model_name=model_name,
                    task=args.task,
                    modalities=modalities,
                    horizon=horizon,
                    lookback=args.lookback,
                    eval_mode=args.eval_mode,
                    news_lag_days=args.news_lag_days,
                    seed=args.seed,
                )
                summary = run_experiment(config=config, panel_bundle=panel_bundle)
                summaries.append(summary)

    summary_frame = pd.DataFrame(
        {
            "run_name": [summary["run_name"] for summary in summaries],
            "fold_count": [summary["fold_count"] for summary in summaries],
            "output_dir": [summary["output_dir"] for summary in summaries],
            **{
                key: [summary["average_metrics"].get(key) for summary in summaries]
                for key in sorted(
                    {
                        metric_name
                        for summary in summaries
                        for metric_name in summary["average_metrics"].keys()
                    }
                )
            },
        }
    )
    summary_frame.to_csv(output_dir / "experiment_grid_summary.csv", index=False)
    (output_dir / "experiment_grid_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
