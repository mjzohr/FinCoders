# Stock Forecast Training

This workspace now includes a reusable training pipeline for three model families:

- `lightgbm`: strong tabular baseline on lagged price features and optional news/sentiment features.
- `lstm`: seminal sequence model with separate price and news encoders.
- `timexer`: a practical TimeXer-style exogenous transformer with patched price tokens and cross-attention over news features.

It also now includes one Hugging Face transformer implementation:

- `hf_patchtst`: Hugging Face `PatchTST`, useful when you want an official transformer-based time-series model from the `transformers` library.

The default horizons are:

- `1` trading day ahead
- `5` trading days ahead
- `21` trading days ahead

## Why these three

- Base model: LightGBM is still one of the hardest classical baselines to beat on structured financial features.
- Seminal model: LSTM remains the canonical sequence model for noisy market time series.
- SOTA-oriented model: TimeXer is designed for time-series forecasting with exogenous variables, which fits your price + news/sentiment setup.

## Data assumptions

- Price source: `data/dates_on_left_stock_data.csv`
- Default news source: `data/news_all_sentiment.csv`
- News is aggregated by ticker and day, then shifted by `--news-lag-days` to reduce leakage when article timestamps are unavailable.
- Targets are derived directly from future prices, not from the existing label bins, so the same code works for next-day, next-week, and next-month forecasting.

## Best-practice choices baked in

- Time-based evaluation only. No random shuffling across dates.
- Walk-forward validation by default.
- Separate training per horizon.
- Regression target is forward return; directional metrics are reported alongside regression metrics.
- Cross-sectional metrics are included:
  - overall Spearman IC
  - average daily Spearman IC
  - top-vs-bottom decile spread
- Missing news is handled explicitly with zeros and a `has_news` feature.

## Install

```bash
pip install -r requirements-training.txt
```

## Train one run

```bash
python -m stock_forecasting.train --model lightgbm --modalities price_news --horizon 1
python -m stock_forecasting.train --model lstm --modalities price_news --horizon 5
python -m stock_forecasting.train --model timexer --modalities price_news --horizon 21
python -m stock_forecasting.train --model hf_patchtst --modalities price_news --horizon 21 --lookback 60
```

## Compare all models and horizons

```bash
python -m stock_forecasting.run_experiments --task regression
```

Or use the bash runner:

```bash
bash run_training_suite.sh
```

Examples:

```bash
PROFILE=competition bash run_training_suite.sh
PROFILE=quick bash run_training_suite.sh
MODELS="lightgbm lstm hf_patchtst" HORIZONS="1 5" bash run_training_suite.sh
LOOKBACKS="30 60" MODALITIES="price_news" bash run_training_suite.sh
INSTALL_DEPS=1 bash run_training_suite.sh
bash run_transformer_suite.sh
```

The bash runner is now designed for experiment comparison, not just single-pass training. The recommended full run for the evaluation notebooks is:

```bash
bash run_training_suite.sh
```

That now defaults to the `competition` profile, which sweeps:

- models: `lightgbm`, `lstm`, `timexer`, `hf_patchtst`
- horizons: `1`, `5`, `21`
- modalities: `price`, `price_news`
- news lags: `1`, `2`
- seeds: `7`, `19`, `31`
- evaluation: `walkforward`
- task: `regression`
- lookbacks by horizon:
  - `1` day horizon: `20`, `30`, `60`
  - `5` day horizon: `30`, `45`, `60`
  - `21` day horizon: `21`, `42`, `60`

This default sweep is intentionally broad so [compare_results.ipynb](C:/Users/zohrabim/Desktop/Projects/stock/compare_results.ipynb) and [horizon_evaluation_report.ipynb](C:/Users/zohrabim/Desktop/Projects/stock/horizon_evaluation_report.ipynb) have enough diversity to compare:

- short vs medium vs long lookback windows
- single-modal vs bimodal models
- sensitivity to news lag assumptions
- which model family wins at each prediction horizon
- whether a result stays strong across repeated random seeds

Useful controls:

- `PROFILE=competition`: recommended full sweep for final model selection
- `PROFILE=quick`: smaller sweep for smoke tests
- `PROFILE=transformers`: transformer-only sweep with `timexer` and `hf_patchtst`
- `LOOKBACKS="30 60"`: override all horizon-specific lookbacks
- `TASKS="regression classification"`: run both tasks, though it is best to compare them separately in the notebook
- `SEEDS="7 19"`: add repeated runs for stability checks
- `SKIP_EXISTING=1`: resume a large sweep without retraining runs that already produced `summary.json`

If you want only transformer-based models, use either of these:

```bash
PROFILE=transformers bash run_training_suite.sh
bash run_transformer_suite.sh
```

The transformer sweep focuses on:

- models: `timexer`, `hf_patchtst`
- modality: `price_news`
- lookbacks:
  - `1` day horizon: `30`, `60`
  - `5` day horizon: `45`, `60`
  - `21` day horizon: `42`, `60`

## Compare results

Open [compare_results.ipynb](C:/Users/zohrabim/Desktop/Projects/stock/compare_results.ipynb) after training finishes. It scans `artifacts/`, ranks runs, compares horizons and modalities, and generates presentation-oriented plots and talking points.

If you want a decision report split explicitly into daily, weekly, and monthly model selection, open [horizon_evaluation_report.ipynb](C:/Users/zohrabim/Desktop/Projects/stock/horizon_evaluation_report.ipynb). It recommends one model per horizon and shows the evidence for why it was chosen.

Both notebooks now include Plotly-based stock-history charts for 5 randomly selected tickers from the evaluated run. They show:

- the historical stock price series
- the actual test-period price path
- the predicted price path

They also now include an additional candlestick-style view. Because the dataset contains closing prices rather than full intraday OHLC data, those candles are explicitly close-derived rather than true market OHLC candles.

The chart frequency follows the trained model horizon:

- daily for `horizon = 1`
- weekly-style aggregation for `horizon = 5`
- monthly-style aggregation for `horizon = 21`

## Save artifacts during experiments

Each experiment directory now saves:

- `config.json`
- `feature_metadata.json`
- `summary.csv`
- `summary.json`
- per-fold `test_predictions.csv`
- per-fold model files:
  - LightGBM: `model.txt`
  - neural models: `model.pt` and `standardizer.npz`

## Train a champion model

After reviewing experiments, retrain one selected configuration into a single deployable artifact under `champions/`.

Choose a specific run:

```bash
python -m stock_forecasting.train_champion --run-name lightgbm_h1_price_news_regression_lb30_lag1_walkforward_s7
```

Or automatically pick the best completed run for a task:

```bash
python -m stock_forecasting.train_champion --task regression
```

Champion artifacts save:

- `config.json`
- `feature_metadata.json`
- `selection.json`
- `champion_summary.json`
- `validation_predictions.csv`
- model file:
  - LightGBM: `model.txt`
  - neural models: `model.pt` and `standardizer.npz`

## Score fresh daily data

Use a saved champion artifact to score the latest available snapshot in your `data/` files:

```bash
python -m stock_forecasting.predict_live --artifact-dir champions/YOUR_CHAMPION_NAME
```

Optional scoring date override:

```bash
python -m stock_forecasting.predict_live --artifact-dir champions/YOUR_CHAMPION_NAME --as-of-date 2024-11-20
```

The live scoring script writes a ranked CSV like `live_predictions_YYYY-MM-DD.csv` inside the champion directory.

## Single-modal vs bimodal

- Single-modal: `--modalities price`
- Bimodal: `--modalities price_news`

## Recommended starting points

- Next day: `lightgbm` and `lstm`
- Next week: `lightgbm`, `lstm`, `timexer`, and `hf_patchtst`
- Next month: `timexer` or `hf_patchtst` with a longer lookback such as `--lookback 60`

## Notes

- The current implementation uses aggregated news and sentiment features rather than a heavy raw-text encoder.
- If you later want full text embeddings, the cleanest extension is to precompute article embeddings and add them as extra exogenous features.
- `hf_patchtst` uses the official Hugging Face PatchTST implementation and concatenates price/news features along the channel dimension.
