#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

torchrun --nproc_per_node 8 -m qh360_vl.eval.model_vqa_pope_llama3 \
    --model-path $INIT_MODEL_PATH/$CKPT \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --slide_window \
    --conv-mode llama3

python qh360_vl/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl