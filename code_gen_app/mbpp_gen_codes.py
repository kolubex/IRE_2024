import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

ds = load_dataset("google-research-datasets/mbpp", "full")

data = torch.load("retrieved_codes.pt")


model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_dir = "/ssd_scratch/advaith/mistral_7b"
print("started loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", cache_dir=model_dir, attn_implementation="flash_attention_2")




def test_function(code_str, test_cases):
    # Execute the code to define the function dynamically
    exec(code_str, globals())
    
    passes = 0
    for test in test_cases:
        try:
            exec(test)  # Use exec() instead of eval() for assertions
            passes += 1
        except AssertionError:
            pass
    
    # Return fraction of passed test cases
    return passes / len(test_cases)



fin_data = {}

filename = "results_R_10k_top5.json"

for item in tqdm(data, desc="generating code with retrieval"):
    task_id = item[0]
    text = item[1]
    test_list = item[2]
    function_name = item[3]
    retrieved_codes = item[4]
    prompt = f"""
    {text} Output only the generated code in python. Do not generate any additional text. Output only the code surrounded by ``` and use this function {function_name}.
    Use these codes for reference: {retrieved_codes}
    """
    encoded_input = tokenizer(prompt, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}
    print("Generating Code...")
    output = model.generate(**encoded_input, max_new_tokens=300)
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Parsing and evaluating code...")
    # remove prompt from generated code
    generated_code = generated_code.replace(prompt, "")
    # get the part between ``` and ```
    match = re.search(r'```(.*?)```', generated_code, re.DOTALL)
    print("Sent to test function...")
    if match:
        code = match.group(1).strip()
    else:
        code = ""  # Fallback if no match is found
    loc_data = {}
    loc_data["input"] = text
    loc_data["output_code"] = code
    loc_data["retrieved_codes"] = retrieved_codes
    fin_data[task_id] = loc_data
    # dump into json file with indent 4
    with open(filename, "w") as f:
        json.dump(fin_data, f, indent=4)


