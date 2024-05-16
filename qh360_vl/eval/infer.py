import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import itertools

from qh360_vl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from qh360_vl.conversation import conv_templates, SeparatorStyle
from qh360_vl.model.builder import load_pretrained_model
from qh360_vl.utils import disable_torch_init
from qh360_vl.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path,process_images_slid_window
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pdb
import sys
from pprint import pprint as pp

g_input_msg = [
    {
        "role": "system", 
        "content": "You are a multilingual, helpful, respectful and honest assistant who can respond in the same language, depending on the language of the question. Try to be as helpful as possible while still being safe. Your answer should not contain anything that is false, unhealthy, harmful, immoral, racist, sexist, toxic, dangerous, or illegal, and if the question relates to such content, please decline to answer. Make sure your answer is socially fair and positive. If a question doesn't make any sense, or is inconsistent with the facts, explain why instead of answering the wrong answer. If you don't know the answer to a question, don't share false information."
    }
]


def get_input(tokenizer, image_processor, model_config, rounds, query, args):
        g_input_msg.append({
            "role": "user", 
            "content": ("<|reserved_special_token_44|>"+ '\n' if not rounds else "") + query
        })
        
        input_ids = tokenizer.apply_chat_template(
            g_input_msg,
            add_generation_prompt=True,
            padding="longest",
            return_tensors="pt",
        )
        input_id_list = input_ids[0].tolist()
        input_id_list[input_id_list.index(128049)]=-200
        input_ids = torch.tensor(input_id_list, dtype=input_ids.dtype,device=input_ids.device)

        image = Image.open(args.image_path).convert('RGB')
        if args.slide_window:
            image_tensor = process_images_slid_window(image, image_processor, model_config, None, None, 336)
        else:
            image_tensor = process_images([image], image_processor, model_config)[0]

        return input_ids.unsqueeze(0), image_tensor.unsqueeze(0)

    
def infer_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    rounds = 0
    while 1:
        try:
            query = input("user: ")
            if query == "exit":
                break
        except:
            continue
            
        input_ids, image_tensor = get_input(tokenizer, image_processor, model.config, rounds, query, args)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                eos_token_id=[tokenizer.convert_tokens_to_ids("<|eot_id|>",)],
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        print("qh360_vl:", outputs)
        
        g_input_msg.append({
            "role": "assistant", 
            "content": outputs
        })
        rounds += 1
        

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--slide_window", action="store_true")
    args = parser.parse_args()
    
    infer_model(args)