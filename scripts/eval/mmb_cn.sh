#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

for SPLIT in {"mmbench_dev_cn_20231003","mmbench_test_cn_20231003",}
do
    torchrun --nproc_per_node 8 -m qh360_vl.eval.model_vqa_mmbench_llama3 \
        --model-path $INIT_MODEL_PATH/$CKPT \
        --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$CKPT.jsonl \
        --single-pred-prompt \
        --temperature 0 \
        --slide_window \
        --lang cn \
        --conv-mode llama3 \

    mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

    python scripts/convert_mmbench_for_submission.py \
        --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
        --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
        --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
        --experiment $CKPT
done