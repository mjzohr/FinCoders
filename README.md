# Stock Forecasting Training Workspace

This repository contains a stock forecasting experimentation workspace built around a reusable Python training pipeline. It supports multiple model families, time-based evaluation, experiment tracking, champion model retraining, and live scoring against the latest available data snapshot.

The code is organized under `stock_forecasting/`, with supporting data files in `data/`, generated outputs in `artifacts/`, and analysis notebooks in the repository root.

## Supported Models

The training pipeline currently supports the following model options:

- `lightgbm`: Gradient-boosted decision tree baseline for structured lagged features.
- `lstm`: Recurrent neural network with separate price and news feature streams.
- `timexer`: Transformer-style sequence model for price data with optional exogenous news features.
- `hf_patchtst`: Hugging Face PatchTST-based time-series model.

## Supported Tasks and Horizons

- Tasks:
  - `regression`
  - `classification`
- Default forecast horizons:
  - `1` trading day
  - `5` trading days
  - `21` trading days

The default training configuration is oriented toward regression experiments with walk-forward evaluation.

## Data Inputs

By default, the workspace expects the following input files:

- Price data: `data/dates_on_left_stock_data.csv`
- News and sentiment data: `data/news_all_sentiment.csv`

When `price_news` mode is used, news features are aggregated by ticker and date, then shifted by `--news-lag-days` to reduce leakage risk when precise article timestamps are not available.

## Evaluation Approach

The training pipeline uses time-based splits only. Random shuffling is not used for model evaluation.

Supported evaluation modes:

- `walkforward` (default)
- `holdout`

For regression runs, the reported metrics include:

- MAE
- RMSE
- directional accuracy
- Spearman information coefficient
- Pearson correlation
- mean daily Spearman information coefficient
- top-bottom decile spread

For classification runs, the reported metrics include:

- accuracy
- balanced accuracy
- precision
- recall
- F1 score

## Installation

Install the required Python packages with:

```bash
pip install -r requirements-training.txt
```

The repository does not currently define a packaged installation step; the command-line examples below assume you are running from the repository root.

## Training a Single Experiment

Examples:

```bash
python -m stock_forecasting.train --model lightgbm --modalities price_news --horizon 1
python -m stock_forecasting.train --model lstm --modalities price_news --horizon 5
python -m stock_forecasting.train --model timexer --modalities price_news --horizon 21
python -m stock_forecasting.train --model hf_patchtst --modalities price_news --horizon 21 --lookback 60
```

Useful training arguments include:

- `--model`: `lightgbm`, `lstm`, `timexer`, or `hf_patchtst`
- `--task`: `regression` or `classification`
- `--modalities`: `price` or `price_news`
- `--horizon`: `1`, `5`, or `21`
- `--lookback`: sequence length used for feature construction
- `--news-lag-days`: lag applied to aggregated news features
- `--eval-mode`: `walkforward` or `holdout`

## Running Experiment Grids

To run a Python-managed experiment grid:

```bash
python -m stock_forecasting.run_experiments --task regression
```

This command writes per-run outputs under `artifacts/` and also produces consolidated experiment summaries.

## Shell-Based Training Sweeps

The repository also includes Bash helpers for broader experiment sweeps:

- `run_training_suite.sh`
- `run_transformer_suite.sh`

Example usage on a Unix-like shell:

```bash
bash run_training_suite.sh
PROFILE=quick bash run_training_suite.sh
PROFILE=transformers bash run_training_suite.sh
bash run_transformer_suite.sh
```

The default `run_training_suite.sh` profile is `competition`, which expands to a broader sweep across:

- models
- horizons
- modalities
- news lags
- seeds
- horizon-specific lookback windows

Additional environment variables supported by the script include:

- `MODELS`
- `HORIZONS`
- `MODALITIES`
- `TASKS`
- `NEWS_LAGS`
- `SEEDS`
- `LOOKBACKS`
- `SKIP_EXISTING`
- `INSTALL_DEPS`

On Windows, the Python module entry points are the more portable option unless you are running these scripts inside a Bash-compatible environment.

## Experiment Artifacts

Each experiment run is written to its own directory under `artifacts/`. Depending on model type, outputs include:

- `config.json`
- `feature_metadata.json`
- `summary.csv`
- `summary.json`
- per-fold `test_predictions.csv`
- model files:
  - `model.txt` for LightGBM
  - `model.pt` and `standardizer.npz` for neural models

The run naming convention is derived from model, horizon, modality, task, lookback, news lag, evaluation mode, and seed.

## Champion Model Retraining

After reviewing experiment results, you can retrain a selected configuration into a deployable artifact under `champions/`.

Examples:

```bash
python -m stock_forecasting.train_champion --run-name lightgbm_h1_price_news_regression_lb30_lag1_walkforward_s7
python -m stock_forecasting.train_champion --task regression
```

Champion artifacts include:

- `config.json`
- `feature_metadata.json`
- `selection.json`
- `champion_summary.json`
- `validation_predictions.csv`
- model file outputs appropriate to the selected model family

## Live Scoring

Use a saved champion artifact to score the latest available data snapshot:

```bash
python -m stock_forecasting.predict_live --artifact-dir champions/YOUR_CHAMPION_NAME
```

Optional date override:

```bash
python -m stock_forecasting.predict_live --artifact-dir champions/YOUR_CHAMPION_NAME --as-of-date 2024-11-20
```

The scoring command writes a ranked CSV file inside the selected champion directory by default.

## Notebooks

The repository includes analysis notebooks in the project root, including:

- `compare_results.ipynb`
- `horizon_evaluation_report.ipynb`

These notebooks are intended for post-training comparison and reporting based on the contents of `artifacts/`.

## Repository Structure

```text
FinCoders/
|-- artifacts/
|-- data/
|-- stock_forecasting/
|-- compare_results.ipynb
|-- horizon_evaluation_report.ipynb
|-- requirements-training.txt
|-- run_training_suite.sh
|-- run_transformer_suite.sh
`-- README.md
```

## Notes

- The current implementation uses aggregated news and sentiment features rather than raw-text language model embeddings.
- GPU execution is supported when `torch` detects CUDA and `--device cuda` is used; otherwise training falls back to CPU.
- The default command-line device setting is `cuda`, but the training code automatically uses CPU when CUDA is unavailable.
