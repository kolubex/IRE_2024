import os
#os.chdir("IRE_fin")


from groq import Groq
import time

import json
from tqdm import tqdm

with open("gold_codes_difficult.json") as f:
    data = json.load(f)


client = Groq(
    api_key= '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',
)
final_data = []
cnt = 0
for item in tqdm(data, total=len(data)):
    question = item["question"]
    task_id = item["task_id"]
    golden_code = item["golden_code"]
    prompt = f"""
    Question: Given this code: {golden_code}, generate a short 1-2 line high level description of the code. Just describe the functionality of the code
    in 2-3 lines. Do not generate any additional text. Only ouput the functionality of the code.
    """
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="llama-3.1-8b-instant",
    temperature=0.0,
    )
    descr = chat_completion.choices[0].message.content
    item["description"] = descr
    cnt+=1
    if cnt % 30 == 0:
        # sleep for a minniute
        time.sleep(60)
    filename = "gold_codes_difficult.json"
    # usei ndent 4
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    
    

    

