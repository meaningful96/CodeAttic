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
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--train-path-dict "$DATA_DIR/train_bfs.pkl" \
--valid-path-dict "$DATA_DIR/valid_bfs.pkl" \
--shortest-path "$DATA_DIR/train_st.pkl" \
--degree-train "${DATA_DIR}/degree_train.json" \
--degree-valid "${DATA_DIR}/degree_valid.json" \
--task ${TASK} \
--batch-size 3072 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 1536 \
--finetune-t \
--finetune-B \
--B 10 \
--pre-batch 0 \
--epochs 20 \
--workers 8 \
--max-to-keep 5 "$@"
