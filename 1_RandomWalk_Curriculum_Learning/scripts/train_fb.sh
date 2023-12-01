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
--linkGraph-path "$DATA_DIR/linkGraph.txt.json" \
--task ${TASK} \
--batch-size 1024 \
--negative-size 64 \
--print-freq 100 \
--additive-margin 0.02 \
--use-self-negative \
--pre-batch 2 \
--randomwalk-step 4 \
--iteration 300 \
--use-amp \
--finetune-t \
--epochs  15 \
--workers 4 \
--max-to-keep 5 "$@"
