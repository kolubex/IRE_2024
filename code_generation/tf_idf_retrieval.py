import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np


from datasets import load_dataset
import re
import json

filename = "descriptions_mistral.json"
import json
with open(filename
            , 'r') as file:
        data = json.load(file)


descr_dict  = {}
for item in data.keys():
    loc_dict = data[item]
    description = loc_dict['description']
    descr_dict[item] = description

ds = load_dataset("google-research-datasets/mbpp", "full")
data = ds['test']

code_map = {}
for item in data:
    task_id = item['task_id']
    code = item['code']
    code_map[task_id] = code

code_db = []
docs = []
for item in descr_dict.keys():
    description = descr_dict[item]
    code = code_map[int(item)]
    row = [description, code]
    docs.append(description)
    code_db.append(row)



# build a tf-idf vectorizzer using the top 1000 tokens 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(docs)


vectorized_docs = tfidf_matrix.toarray()
# convert to numpy array
vectorized_docs = np.array(vectorized_docs)
print(vectorized_docs.shape)

vectorized_db = []
for row in code_db:
    code = row[1]
    descr = row[0]
    vector = vectorizer.transform([descr]).toarray()
    vectorized_db.append([vector, code])

import torch
from tqdm import tqdm


import re
fin_data = []
for item in tqdm(data, total=len(data)):
    task_id = item['task_id']
    text = item['text']
    test_list = item['test_list']
    code = code_map[task_id]
    function_name = re.search(r'def(.*?)\:', code, re.DOTALL)
    function_name = function_name.group(1)
    function_name = f"def {function_name.strip()}:"
    retrieved_codes = []
    vectorized_text = vectorizer.transform([text]).toarray()
    all_codes = []
    for row in vectorized_db:
        vector = row[0]
        code = row[1]
        sim = cosine_similarity(vector, vectorized_text)
        all_codes.append([sim, code])
    # take top 3 codes with top sim and add to retrieved_codes
    all_codes = sorted(all_codes, key=lambda x: x[0], reverse=True)
    for i in range(3):
        retrieved_codes.append(all_codes[i][1])
    fin_data.append([task_id, text, test_list, function_name, retrieved_codes])


filename = "retrieved_codes.pt"
torch.save(fin_data, filename)

    