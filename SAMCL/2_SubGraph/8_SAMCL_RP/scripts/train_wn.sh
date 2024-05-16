#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--train-path-dict "${DATA_DIR}/train_antithetical_50_400.pkl" \
--valid-path-dict "${DATA_DIR}/valid_antithetical_20_300.pkl" \
--shortest-path "${DATA_DIR}/ShortestPath_train.pkl" \
--degree-train "${DATA_DIR}/Degree_train.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid.pkl"  \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 512 \
--finetune-t \
--finetune-B \
--B 10e3 \
--epochs 50 \
--workers 4 \
--max-to-keep 10 "$@"

