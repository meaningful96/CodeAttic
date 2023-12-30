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
--appearance-path "$DATA_DIR/appearance/fb512.json" \
--task ${TASK} \
--batch-size 3072 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--subgraph-size 1536 \
--finetune-t \
--num-iter 500 \
--k-steps 20 \
--num-process 50 \
--pre-batch 0 \
--epochs 20 \
--workers 4 \
--max-to-keep 5 "$@"
