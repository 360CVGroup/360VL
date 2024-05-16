#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"
CKPT="qh360_vl-8B"

python -m qh360_vl.eval.infer \
    --model-path $INIT_MODEL_PATH/$CKPT \
    --image-path /hbox2dir/test.jpg \
    --slide_window
