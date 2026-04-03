#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-transformers}"
MODELS="${MODELS:-timexer hf_patchtst}"
MODALITIES="${MODALITIES:-price_news}"
export PROFILE MODELS MODALITIES

bash run_training_suite.sh "$@"
