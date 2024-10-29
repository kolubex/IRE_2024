import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
import re
import json
def load_mbpp():
    return load_dataset("google-research-datasets/mbpp", "full")

ds = load_mbpp()

print(ds.shape)

data = ds["test"]


## loading the model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_dir = "/ssd_scratch/advaith/mistral_7b"
print("started loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", cache_dir=model_dir, attn_implementation="flash_attention_2")


fin_data = {}
json_res = "descriptions_mistral.json"
for task in tqdm(data, desc = "Generating Descriptions"):

    code = task["code"]
    prompt = f"""
    For this snippet of code: {code} generate a brief high level two-three line description that effectively captures the overall functionality of the provided code. Output only the description of the code. Do not generate any additional text. Do not make the description too long or detailed.
    """

    encoded_input = tokenizer(prompt, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    outputs = model.generate(
    **encoded_input, max_new_tokens = 500
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sample_id = task["task_id"]
    # replace the prompt in generated text with ""
    generated_text = generated_text.replace(prompt, "")
    # remove leading and trailing whitespaces
    generated_text = generated_text.strip()
    loc_dict = {}
    loc_dict["description"] = generated_text
    loc_dict["prompt"] = prompt
    fin_data[sample_id] = loc_dict
    # dump into json file with indent 4
    with open(json_res, "w") as f:
        json.dump(fin_data, f, indent=4)
    
