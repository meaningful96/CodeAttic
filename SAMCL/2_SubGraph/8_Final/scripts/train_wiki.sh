#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_ind"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    TASK=$1
    shift
fi

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main_wiki.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--train-path-dict "${DATA_DIR}/train_antithetical_60_300.pkl" \
--valid-path-dict "${DATA_DIR}/valid_antithetical_60_300.pkl" \
--shortest-path "${DATA_DIR}/ShortestPath_train.pkl" \
--degree-train "${DATA_DIR}/Degree_train.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid.pkl" \
--task "${TASK}" \
--batch-size 1024 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 512 \
--finetune-t \
--finetune-B \
--B 1000 \
--epochs 1 \
--workers 5 \
--max-to-keep 10 "$@"
