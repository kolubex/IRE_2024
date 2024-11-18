import os
os.chdir("IRE_fin")

import json
from pprint import pprint
import numpy as np
import re
from tqdm import tqdm
gold_file = "data/gold_codes_difficult.json"
gold_data = json.load(open(gold_file))


test_info = {}
for item in gold_data:
    task_id = item["task_id"]
    test_list = item["test_list"]
    test_function = item["test_function"]
    entry_point = item["entry_point"]
    test_matching = item["test_matching"]
    test_match_function = item["test_match_function"]
    golden_code = item["golden_code"]
    loc_dict = {}
    loc_dict["test_list"] = test_list
    loc_dict["test_function"] = test_function
    loc_dict["entry_point"] = entry_point
    loc_dict["test_matching"] = test_matching
    loc_dict["test_match_function"] = test_match_function
    loc_dict["golden_code"] = golden_code
    test_info[task_id] = loc_dict

filename = "data/difficult_results.json"
with open(filename, "r") as f:
    data = json.load(f)

def check_sanity(code, test_function):
    code = code + "\n" + test_function
    try:
        exec(code)
        return 1
    except:
        return 0
def evaluate_code(code, test_function, test_list, entry_point):
    test_list = [item.replace("candidate", entry_point) for item in test_list]
    code = code + "\n" + test_function
    exec(code)
    pass_list = []
    for test in test_list:
        try:
            exec(test)
            pass_list.append(1)
        except:
            pass_list.append(0)
    return np.mean(pass_list)

no_ret_no_plan_compile = []
no_ret_no_plan_pass_k = []
ret_no_plan_compile = []
ret_no_plan_pass_k = []
no_ret_plan_compile = []
no_ret_plan_pass_k = []
ret_plan_compile = []
ret_plan_pass_k = []
golden_compile = []
golden_pass_k = []
for item in tqdm(data, total = len(data)):
    task_id = item["task_id"]
    test_dict = test_info[task_id]
    test_function = test_dict["test_function"]
    test_list = test_dict["test_list"]
    entry_point = test_dict["entry_point"]
    code = item['vanilla_code']
    if not check_sanity(code, test_function):
        no_ret_no_plan_compile.append(0)
        no_ret_no_plan_pass_k.append(0)
    if check_sanity(code, test_function):
        no_ret_no_plan_compile.append(1)
        no_ret_no_plan_pass_k.append(evaluate_code(code, test_function, test_list, entry_point))
    code = item["code_with_retrieval"]
    if not check_sanity(code, test_function):
        ret_no_plan_compile.append(0)
        ret_no_plan_pass_k.append(0)
    if check_sanity(code, test_function):
        ret_no_plan_compile.append(1)
        ret_no_plan_pass_k.append(evaluate_code(code, test_function, test_list, entry_point))
    code = item["code_with_plan"]
    if not check_sanity(code, test_function):
        no_ret_plan_compile.append(0)
        no_ret_plan_pass_k.append(0)
    if check_sanity(code, test_function):
        no_ret_plan_compile.append(1)
        no_ret_plan_pass_k.append(evaluate_code(code, test_function, test_list, entry_point))
    code = item["code_with_planning_and_retrieval"]
    if not check_sanity(code, test_function):
        ret_plan_compile.append(0)
        ret_plan_pass_k.append(0)
    if check_sanity(code, test_function):
        ret_plan_compile.append(1)
        ret_plan_pass_k.append(evaluate_code(code, test_function, test_list, entry_point))
    code = test_dict["golden_code"]
    if not check_sanity(code, test_function):
        golden_compile.append(0)
        golden_pass_k.append(0)
    if check_sanity(code, test_function):
        golden_compile.append(1)
        golden_pass_k.append(evaluate_code(code, test_function, test_list, entry_point))
        

    
    print("pass@k for code that compiles without retrieval and planning: ", np.mean(no_ret_no_plan_pass_k))
print("pass@k for code that compiles with retrieval and without planning: ", np.mean(ret_no_plan_pass_k))
print("pass@k for code that compiles without retrieval and with planning: ", np.mean(no_ret_plan_pass_k))
print("pass@k for code that compiles with retrieval and planning: ", np.mean(ret_plan_pass_k))