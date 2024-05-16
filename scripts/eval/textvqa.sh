#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

torchrun --nproc_per_node 8 -m qh360_vl.eval.model_vqa_loader_llama3 \
    --model-path $INIT_MODEL_PATH/$CKPT \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --slide_window \
    --conv-mode llama3

python -m qh360_vl.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl