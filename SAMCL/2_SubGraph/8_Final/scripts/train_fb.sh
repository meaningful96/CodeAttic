#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"

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
--pretrained-model microsoft/deberta-large \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--train-path-dict "$DATA_DIR/train_antithetical_40_300.pkl" \
--valid-path-dict "$DATA_DIR/valid_antithetical_40_300.pkl" \
--shortest-path "$DATA_DIR/ShortestPath_train.pkl" \
--degree-train "${DATA_DIR}/Degree_train.pkl" \
--degree-valid "${DATA_DIR}/Degree_valid.pkl" \
--task ${TASK} \
--batch-size 768 \
--print-freq 50 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 384 \
--finetune-t \
--finetune-B \
--B 1000 \
--num-iter 500 \
--k-steps 10 \
--epochs 20 \
--workers 4 \
--max-to-keep 5 "$@"
