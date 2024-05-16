INIT_MODEL_PATH="/hbox2dir"

name="qh360_vl-llama3-70B"

python -m qh360_vl.eval.model_vqa_loader_llama3_nodist \
    --model-path $INIT_MODEL_PATH/$name \
    --question-file custom/vqa_test_custom.jsonl \
    --image-folder custom/vqa \
    --answers-file custom/$name.jsonl \
    --temperature 0 \
    --slide_window \
    --conv-mode llama3
