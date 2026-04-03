#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-competition}"
MODELS="${MODELS:-lightgbm lstm timexer hf_patchtst}"
HORIZONS="${HORIZONS:-1 5 21}"
MODALITIES="${MODALITIES:-price price_news}"
TASKS="${TASKS:-regression}"
NEWS_LAGS="${NEWS_LAGS:-1 2}"
SEEDS="${SEEDS:-7}"
EVAL_MODE="${EVAL_MODE:-walkforward}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
PRICE_CSV="${PRICE_CSV:-data/dates_on_left_stock_data.csv}"
NEWS_CSV="${NEWS_CSV:-data/news_all_sentiment.csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
LOOKBACKS="${LOOKBACKS:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ "${PROFILE}" == "transformers" ]]; then
  if [[ "${MODELS}" == "lightgbm lstm timexer hf_patchtst" ]]; then
    MODELS="timexer hf_patchtst"
  fi
  if [[ "${MODALITIES}" == "price price_news" ]]; then
    MODALITIES="price_news"
  fi
fi

if [[ "${PROFILE}" == "competition" ]]; then
  if [[ "${SEEDS}" == "7" ]]; then
    SEEDS="7 19 31"
  fi
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install -r requirements-training.txt
fi

default_lookbacks_for_horizon() {
  local horizon="$1"
  if [[ -n "${LOOKBACKS}" ]]; then
    echo "${LOOKBACKS}"
    return
  fi

  case "${PROFILE}:${horizon}" in
    competition:1) echo "20 30 60" ;;
    competition:5) echo "30 45 60" ;;
    competition:21) echo "21 42 60" ;;
    transformers:1) echo "30 60" ;;
    transformers:5) echo "45 60" ;;
    transformers:21) echo "42 60" ;;
    quick:1) echo "20 30" ;;
    quick:5) echo "30 45" ;;
    quick:21) echo "21 42" ;;
    comprehensive:1) echo "20 30 60" ;;
    comprehensive:5) echo "30 45 60" ;;
    comprehensive:21) echo "21 42 60" ;;
    *:1) echo "20 30 60" ;;
    *:5) echo "30 45 60" ;;
    *:21) echo "21 42 60" ;;
    *) echo "30 60" ;;
  esac
}

planned_runs=0
for task in ${TASKS}; do
  for modality in ${MODALITIES}; do
    for horizon in ${HORIZONS}; do
      for lookback in $(default_lookbacks_for_horizon "${horizon}"); do
        for news_lag in ${NEWS_LAGS}; do
          for seed in ${SEEDS}; do
            for model in ${MODELS}; do
              planned_runs=$((planned_runs + 1))
            done
          done
        done
      done
    done
  done
done

echo "Training configuration"
echo "  profile: ${PROFILE}"
echo "  models: ${MODELS}"
echo "  horizons: ${HORIZONS}"
echo "  modalities: ${MODALITIES}"
echo "  tasks: ${TASKS}"
echo "  news_lags: ${NEWS_LAGS}"
echo "  seeds: ${SEEDS}"
echo "  eval_mode: ${EVAL_MODE}"
echo "  output_dir: ${OUTPUT_DIR}"
echo "  skip_existing: ${SKIP_EXISTING}"
echo "  planned_runs: ${planned_runs}"
echo

for task in ${TASKS}; do
  for modality in ${MODALITIES}; do
    for horizon in ${HORIZONS}; do
      for lookback in $(default_lookbacks_for_horizon "${horizon}"); do
        for news_lag in ${NEWS_LAGS}; do
          for seed in ${SEEDS}; do
            for model in ${MODELS}; do
              run_name="${model}_h${horizon}_${modality}_${task}_lb${lookback}_lag${news_lag}_${EVAL_MODE}_s${seed}"
              if [[ "${SKIP_EXISTING}" == "1" && -f "${OUTPUT_DIR}/${run_name}/summary.json" ]]; then
                echo "=== Skipping ${run_name} (summary.json already exists) ==="
                echo
                continue
              fi
              echo "=== Running ${run_name} ==="
              "${PYTHON_BIN}" -m stock_forecasting.train \
                --price-csv "${PRICE_CSV}" \
                --news-csv "${NEWS_CSV}" \
                --output-dir "${OUTPUT_DIR}" \
                --model "${model}" \
                --task "${task}" \
                --modalities "${modality}" \
                --horizon "${horizon}" \
                --lookback "${lookback}" \
                --news-lag-days "${news_lag}" \
                --eval-mode "${EVAL_MODE}" \
                --seed "${seed}"
              echo
            done
          done
        done
      done
    done
  done
done

echo "All runs completed. Summaries are under ${OUTPUT_DIR}/"
