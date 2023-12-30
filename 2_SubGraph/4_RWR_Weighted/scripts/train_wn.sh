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
--train-path-dict "${DATA_DIR}/train_uniform20_500.npy" \
--valid-path-dict "${DATA_DIR}/valid_uniform20_500.npy" \
--appearance-path "${DATA_DIR}/appearance/wn512.json" \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--pre-batch 0 \
--subgraph-size 512 \
--k-steps 20 \
--num-iter 2000 \
--num-process 30 \
--finetune-t \
--epochs 50 \
--workers 4 \
--max-to-keep 3 "$@"
