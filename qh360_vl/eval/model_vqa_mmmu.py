import re
import random
import numpy as np
import os
import json
import yaml
import torch

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser

from qh360_vl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from qh360_vl.conversation import conv_templates, SeparatorStyle
from qh360_vl.model.builder import load_pretrained_model
from qh360_vl.utils import disable_torch_init
from qh360_vl.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path ,process_images_slid_window

CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}


# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f'({can})')
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def call_bunny_engine_df(args, sample, model, tokenizer=None, processor=None):
    prompt = sample['final_input_prompt']
    input_msg = [
        {
            "role": "system", 
#             "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                "content": "You are a multilingual, helpful, respectful and honest assistant who can respond in the same language, depending on the language of the question. Try to be as helpful as possible while still being safe. Your answer should not contain anything that is false, unhealthy, harmful, immoral, racist, sexist, toxic, dangerous, or illegal, and if the question relates to such content, please decline to answer. Make sure your answer is socially fair and positive. If a question doesn't make any sense, or is inconsistent with the facts, explain why instead of answering the wrong answer. If you don't know the answer to a question, don't share false information."
        },
        {
            "role": "user", 
            "content": "<|reserved_special_token_44|>"+ '\n' + prompt
        }
    ]
    input_ids = tokenizer.apply_chat_template(
        input_msg,
        add_generation_prompt=True,
        padding="longest",
        return_tensors="pt",
    )
    # print(input_ids)
    input_id_list = input_ids[0].tolist()
    input_id_list[input_id_list.index(128049)]=-200
    input_ids = torch.tensor(input_id_list, dtype=input_ids.dtype,device=input_ids.device).unsqueeze(0)

    image = sample['image'].unsqueeze(0)
    terminators = [
            tokenizer.convert_tokens_to_ids("<|eot_id|>",)
    ]
    if image is not None:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=False,
                temperature=0,
                top_p=None,
                eos_token_id=terminators,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        response = outputs.strip()
        print(outputs)
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict


def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
                'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
                'image': data['image_1'], 'question_type': data['question_type']}


# DATA PROCESSING
def construct_prompt(sample, config):
    question = sample['question']
    options = eval(sample['options'])
    example = ""
    if sample['question_type'] == 'multiple-choice':
        start_chr = 'A'
        prediction_range = []
        index2ans = {}
        for option in options:
            prediction_range.append(start_chr)
            example += f"({start_chr}) {option}\n"
            index2ans[start_chr] = option
            start_chr = chr(ord(start_chr) + 1)
        empty_prompt_sample_structure = config['multi_choice_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question, example)
        res_dict = {}
        res_dict['index2ans'] = index2ans
        res_dict['correct_choice'] = sample['answer']
        res_dict['all_choices'] = prediction_range
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt

        res_dict['gt_content'] = options[ord(sample['answer'].upper()) - ord('A')]
    else:
        empty_prompt_sample_structure = config['short_ans_example_format']
        empty_prompt = empty_prompt_sample_structure.format(question)
        res_dict = {}
        res_dict['empty_prompt'] = empty_prompt
        if config['task_instructions']:
            res_dict['final_input_prompt'] = config['task_instructions'].strip() + '\n\n' + empty_prompt
        else:
            res_dict['final_input_prompt'] = empty_prompt
        res_dict['gt_content'] = sample['answer']

    res_dict.update(sample)
    return res_dict


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            if args.small_gpu_usage:
                sample['image'] = sample['image'].cuda()
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)
            if args.small_gpu_usage:
                sample['image'] = sample['image'].cpu()

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--model-base', type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--small-gpu-usage", action="store_true")
    parser.add_argument("--slide_window", action="store_true")
    
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    processor = None
    call_model_engine = call_bunny_engine_df # ?????

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    # load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.pad_token = tokenizer.eos_token

    samples = []
    for i, sample in enumerate(tqdm(dataset)):
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, args.config)
        if sample['image']:
            if args.small_gpu_usage:
                if not args.slide_window:
                    sample['image'] = process_images([sample['image'].convert("RGB")], image_processor, model.config)[0]
                else:
                    sample['image'] = process_images_slid_window(sample['image'].convert("RGB"), image_processor, model.config, None, None, 336)
            else:
                if not args.slide_window:
                    sample['image'] = process_images([sample['image'].convert("RGB")], image_processor, model.config)[0].to(device)
                else:
                    sample['image'] = process_images_slid_window(sample['image'].convert("RGB"), image_processor, model.config, None, None, 336).to(device)

        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, 'w') as f:
        json.dump(out_samples, f, indent=4)


if __name__ == '__main__':
    main()