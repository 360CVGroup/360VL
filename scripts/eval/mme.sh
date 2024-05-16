#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

torchrun --nproc_per_node 8 -m qh360_vl.eval.model_vqa_loader_llama3 \
    --model-path $INIT_MODEL_PATH/$CKPT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --slide_window \
    --conv-mode llama3

cd ./playground/data/eval/MME
python convert_answer_to_mme.py --experiment $CKPT

cd eval_tool
python calculation.py --results_dir answers/$CKPT