import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import itertools

from qh360_vl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from qh360_vl.conversation import conv_templates, SeparatorStyle
from qh360_vl.model.builder import load_pretrained_model
from qh360_vl.utils import disable_torch_init
from qh360_vl.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path,process_images_slid_window

from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def eval_model(args):
    # Model

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.pad_token = tokenizer.eos_token

    questions = pd.read_table(os.path.expanduser(args.question_file))
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = get_chunk(questions, int(os.getenv('WORLD_SIZE', '1')), torch.distributed.get_rank())
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
    
    all_outputs = []
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            image = load_image_from_base64(row['image'])
            if not is_none(hint):
                question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question

            qs = "<|reserved_special_token_44|>" + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            input_msg = [
                {
                    "role": "system", 
                    "content": "You are a multilingual, helpful, respectful and honest assistant who can respond in the same language, depending on the language of the question. Try to be as helpful as possible while still being safe. Your answer should not contain anything that is false, unhealthy, harmful, immoral, racist, sexist, toxic, dangerous, or illegal, and if the question relates to such content, please decline to answer. Make sure your answer is socially fair and positive. If a question doesn't make any sense, or is inconsistent with the facts, explain why instead of answering the wrong answer. If you don't know the answer to a question, don't share false information."
                },
                {
                    "role": "user", 
                    "content": qs
                }
            ]
            input_ids = tokenizer.apply_chat_template(
                input_msg,
                add_generation_prompt=True,
                padding="longest",
                return_tensors="pt",
            )
            input_id_list = input_ids[0].tolist()
            input_id_list[input_id_list.index(128049)]=-200
            input_ids = torch.tensor(input_id_list, dtype=input_ids.dtype,device=input_ids.device).unsqueeze(0).cuda()
            if args.slide_window:
                image_tensor = process_images_slid_window(image, image_processor, model.config, None, None, 336)
            else:
                image_tensor = process_images([image], image_processor, model.config)[0]
                
            terminators = [
                tokenizer.convert_tokens_to_ids("<|eot_id|>",)
            ]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    eos_token_id=terminators,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            print(outputs)

            ans_id = shortuuid.uuid()
            all_outputs.append({"question_id": idx,
                                    "round_id": round_idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "options": options,
                                    "option_char": cur_option_char,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}})
            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, all_outputs)

    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        for item in merged_outputs:
            ans_file.write(json.dumps(item) + "\n")
    ans_file.close()
    torch.distributed.barrier()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--slide_window", action="store_true")
    args = parser.parse_args()

    eval_model(args)
