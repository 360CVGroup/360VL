#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

for SPLIT in {"validation","test",}
do
    python -m qh360_vl.eval.model_vqa_mmmu \
        --model-path $INIT_MODEL_PATH/$CKPT \
        --data-path ./playground/data/eval/mmmu/MMMU \
        --config-path ./playground/data/eval/mmmu/config.yaml \
        --output-path ./playground/data/eval/mmmu/answers_upload/$SPLIT/$CKPT.json \
        --split $SPLIT \
        --slide_window \
        --conv-mode llama3
    
    if [[ $SPLIT == "validation" ]]
    then
        python ./playground/data/eval/mmmu/eval_mmmu.py \
            --output-path ./playground/data/eval/mmmu/answers_upload/$SPLIT/$CKPT.json
    fi
done