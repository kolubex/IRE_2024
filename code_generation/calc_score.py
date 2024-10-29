import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import re
import json
def load_mbpp():
    return load_dataset("google-research-datasets/mbpp", "full")

ds = load_mbpp()

print(ds.shape)

data = ds["test"]

filename = "results_R_10k_top5.json"
with open(filename
            , "r") as f:
        results = json.load(f)


test_cases_map = {}
for item in data:
    task_id = item['task_id']
    test_list = item['test_list']
    test_cases_map[task_id] = test_list


import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Execution time exceeded")

def run_test_with_timeout(test, timeout=5):
    # Set the timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the alarm
    
    try:
        exec(test)  # Execute the test code
        return True  # Test passed
    except (AssertionError, TimeoutError):
        return False  # Test failed or timed out
    finally:
        signal.alarm(0)  # Disable the alarm

def test_function(code_str, test_cases):
    # Execute the code to define the function dynamically
    exec(code_str, globals())
    
    passes = 0
    for test in test_cases:
        if run_test_with_timeout(test, timeout=2):  # Run each test with a 5-second timeout
            passes += 1
    
    # Return fraction of passed test cases
    return passes / len(test_cases)


scores = []
for key in tqdm(results.keys(), total=len(results.keys())):
    item = results[str(key)]
    code = item['output_code']
    tests = test_cases_map[int(key)]
    try:
        pass_rate = test_function(code, tests)
    except:
        pass_rate = 0
    scores.append(pass_rate)

    
    
tot = 0
cnt = 0
for i in scores:
    tot += i
    cnt += 1
    
print("Average pass rate: ", tot/cnt)

       