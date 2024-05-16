#!/bin/bash
INIT_MODEL_PATH="/hbox2dir"

name="qh360_vl-llama3-70B"
python -m qh360_vl.eval.model_vqa \
    --model-path $INIT_MODEL_PATH/$name \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --temperature 0 \
    --slide_window \
    --conv-mode llama3

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python qh360_vl/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$name.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl

python qh360_vl/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$name.jsonl