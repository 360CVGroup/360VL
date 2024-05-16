#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

torchrun --nproc_per_node 8 -m qh360_vl.eval.model_vqa_refcoco_llama3 \
    --model-path $INIT_MODEL_PATH/$CKPT \
    --question-file ./playground/data/eval/refcoco/REFCOCO_VAL_en_new.jsonl \
    --image-folder ./playground/data/eval/refcoco/train2014 \
    --answers-file ./playground/data/eval/res_test/$CKPT/refcoco.json \
    --temperature 0 \
    --slide_window \
    --patch_img_size 336 \
    --conv-mode llama3 \
    
python ./qh360_vl/eval/compute_precision.py ./playground/data/eval/res_test/$CKPT/refcoco.json