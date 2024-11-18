import os
os.chdir("IRE_fin")

from code_generator import *
import json
import time
from tqdm import tqdm



code_generator = CodeGen("data/vector_db.pt", "data/gold_codes_difficult.json", model = "llama3-7b")

filepath = "data/gold_codes_difficult.json"
data = json.load(open(filepath))

final_result = []
filename = "data/difficult_results.json"
for row in tqdm(data, total=len(data)):
    question = row["question"]
    task_id = row["task_id"]
    vanilla_code = code_generator.generate_code_retrieve(question, 0)
    if code_generator.global_calls % 30 == 0:
        time.sleep(60)
    code_with_retrieval = code_generator.generate_code_retrieve(question, 2)
    if code_generator.global_calls % 30 == 0:
        time.sleep(60)
    code_with_plan = code_generator.generate_code_with_planning(question, show_planning=False)
    if code_generator.global_calls % 30 == 0:
        time.sleep(60)
    code_with_planning_and_retrieval = code_generator.generate_code_with_planning_retrieval(question, False, 1)
    if code_generator.global_calls % 30 == 0:
        time.sleep(60)
    loc_dict = {}
    loc_dict["task_id"] = task_id
    loc_dict["question"] = question
    loc_dict["vanilla_code"] = vanilla_code
    loc_dict["code_with_retrieval"] = code_with_retrieval
    loc_dict["code_with_plan"] = code_with_plan
    loc_dict["code_with_planning_and_retrieval"] = code_with_planning_and_retrieval
    final_result.append(loc_dict)
    # dump into filename woth indent 4
    json.dump(final_result, open(filename, "w"), indent=4)




