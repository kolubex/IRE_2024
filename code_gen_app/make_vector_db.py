import os
os.chdir("IRE_fin")

import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import torch


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")  

filename = "data/gold_codes_difficult.json"
data = json.load(open(filename))

sent_vector = {}
for item in tqdm(data, total = len(data)):
    description = item["description"]
    encoded_input = tokenizer(description, return_tensors='pt')
    # output last hidden states
    output = model(**encoded_input, output_hidden_states=True)
    last_hidden_states = output.hidden_states[-1]
    # mean of last hidden states
    mean_last_hidden_states = last_hidden_states.mean(dim=1).detach().numpy()
    sent_vector[description] = mean_last_hidden_states

torch.save(sent_vector, "data/vector_db.pt")