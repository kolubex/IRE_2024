import os
os.chdir("IRE_fin")

import json
from tqdm import tqdm

filename = "mbpp_eval/merged_results_mbpp.json"
with open(filename, "r") as f:
    gen_codes = json.load(f)

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
    except (AssertionError, TimeoutError, NameError):
        return False  # Test failed or timed out
    finally:
        signal.alarm(0)  # Disable the alarm

def test_function(code_str, test_cases):
    # remove ` in code
    code_str = code_str.replace("`", "")
    exec(code_str, globals())
    
    passes = 0
    for test in test_cases:
        if run_test_with_timeout(test, timeout=2):  # Run each test with a 5-second timeout
            passes += 1
    
    # Return fraction of passed test cases
    return passes / len(test_cases)


from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "full")

data = ds['test']
test_cases_map = {}
for item in data:
    task_id = item['task_id']
    test_list = item['test_list']
    test_cases_map[task_id] = test_list

vanilla_stats = []
code_with_retrieval_stats = []
code_with_plan_stats = []
code_with_planning_and_retrieval_stats = []
for item in tqdm(gen_codes, total=len(gen_codes)):
    input_id = item['input_id']
    vanilla_code = item['vanilla_code']
    code_with_retrieval = item['code_with_retrieval']
    code_with_plan = item['code_with_plan']
    code_with_planning_and_retrieval = item['code_with_planning_and_retrieval']
    tests = test_cases_map[int(input_id)]
    vanilla_code_pass_rate = test_function(vanilla_code, tests)
    code_with_retrieval_pass_rate = test_function(code_with_retrieval, tests)
    code_with_plan_pass_rate = test_function(code_with_plan, tests)
    code_with_planning_and_retrieval_pass_rate = test_function(code_with_planning_and_retrieval, tests)
    vanilla_stats.append(vanilla_code_pass_rate)
    code_with_retrieval_stats.append(code_with_retrieval_pass_rate)
    code_with_plan_stats.append(code_with_plan_pass_rate)
    code_with_planning_and_retrieval_stats.append(code_with_planning_and_retrieval_pass_rate)

vanilla_score = sum(vanilla_stats) / len(vanilla_stats) 
code_with_retrieval_score = sum(code_with_retrieval_stats) / len(code_with_retrieval_stats)
code_with_plan_score = sum(code_with_plan_stats) / len(code_with_plan_stats)
code_with_planning_and_retrieval_score = sum(code_with_planning_and_retrieval_stats) / len(code_with_planning_and_retrieval_stats)
print("Vanilla score: ", vanilla_score)
print("Code with retrieval score: ", code_with_retrieval_score)
print("Code with plan score: ", code_with_plan_score)
print("Code with planning and retrieval score: ", code_with_planning_and_retrieval_score)
