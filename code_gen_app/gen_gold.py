import os
os.chdir("IRE_fin")


from groq import Groq
client = Groq(
    api_key= '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',
)
import json
from tqdm import tqdm

with open("oops_difficult.json") as f:
    data = json.load(f)

final_data = []
for item in tqdm(data, total=len(data)):
    question = item["question"]
    task_id = item["task_id"]
    prompt = f"""
    Question: {question}
    Generate ONLY the code. Do NOT generate any additional text.
    """
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="mixtral-8x7b-32768",
    )
    gen_code = chat_completion.choices[0].message.content
    item["golden_code"] = gen_code
    item['question'] = question
    item['task_id'] = task_id
    final_data.append(item)

filename = "gold_codes_difficult.json"
# usei ndent 4
with open(filename, "w") as f:
    json.dump(final_data, f, indent=4)
    


