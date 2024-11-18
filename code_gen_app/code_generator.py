import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import torch
from groq import Groq
import re
from torch.nn.functional import cosine_similarity
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")


class CodeGen():
    def __init__(self, vector_db_path, sent_to_code_path, model):
        self.global_calls = 0
        self.api_key = "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"
        self.model = "llama3-70b-8192"
        self.sent_to_code_path = sent_to_code_path
        self.vector_db = torch.load(vector_db_path)
        self.sent_to_code = self.get_sent_to_code()
        self.client = Groq(
            api_key= self.api_key,
        )
    def get_sent_to_code(self):
        filename = self.sent_to_code_path
        data = json.load(open(filename))
        sent_to_code = {}
        for item in data:
            code = item['golden_code']
            sent = item['description']
            sent_to_code[sent] = code
        return sent_to_code
    def retrieve_top_k(self, query, k):
        rankings = {}
        encoded_input = tokenizer(query, return_tensors='pt')
        output = model(**encoded_input, output_hidden_states=True)
        last_hidden_states = output.hidden_states[-1]
        query_vec = last_hidden_states.mean(dim=1).detach().numpy()
        for sent in self.vector_db:
            vector = self.vector_db[sent]
            score = torch.cosine_similarity(torch.tensor(query_vec), torch.tensor(vector), dim=1).item()
            rankings[sent] = score
        return sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:k]
    def clean_code(self, code, ref_codes):
        if "python" in code:
            # replace python with ""
            code = code.replace("python", "")
            code = code.split("```")[1]
        for item in ref_codes:
            code = code.replace(item, "")
        # same string is there twice, remove it
        s = code
        code = s[:len(s)//2] if len(s) % 2 == 0 and s[:len(s)//2] == s[len(s)//2:] else s
        return code
    def generate_text(self, prompt):
        chat_completion = self.client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=self.model,
        temperature=0.0,
        max_tokens = 800
        )
        self.global_calls += 1
        return chat_completion.choices[0].message.content
    def generate_code_retrieve(self, query, k):
        top_k = self.retrieve_top_k(query, k)
        codes = [self.sent_to_code[sent] for sent, _ in top_k]
        if len(codes) == 0:
            prompt = f"""
            Generate code for the following query: {query}.
            Only output the generated code. Do not output any additional information or additional text.
            Output only the required code. Do not generate any additional code, text or example usage information.
            """
        else:
            prompt = f"""
            Generate code for the following query: {query}.
            Make use of these codes as reference while generating the code {codes}. Use the above codes only as reference. Do not copy the code as it is.
            Only output the generated code. Do not output any additional information or additional text.
            Output only the required code. Do not generate any additional code, text or example usage information.
            """
        gen_code = self.generate_text(prompt)
        cleaned_code = self.clean_code(gen_code, codes)
        return cleaned_code
    def generate_plan(self, query):
        prompt = f"""
        Given this query: {query}, generate a plan which outlines the steps to solve the problem. This plan should be in such a
        format that generating code for each step of the plan should generate the final code for the query. Output the plan with each
        step seperated by a "||". Output only the plan. Do not output any additional information or additional text. In the plan, do not 
        include any steps to test the generated code. Your plan should only include the steps to generate the code, modularly.

        """
        plan = self.generate_text(prompt)
        return plan
    def generate_code_with_planning(self, query, show_planning):
        plan = self.generate_plan(query)
        steps = plan.split("||")
        code = ""
        steps_cnt = 1
        for step in steps:
            prompt = f"""
            You are generating code based on a plan. Currently, you have to generate code for the following step: {step}.
            Until now, this code has been generated: {code}. Use the code generated until now to generate the code for the current step.
            Output the final code based on the previous code and by generating code for the current step.
            Output the final code upto the current step only. Do not output any additional information or additional text.
            Output only the required code. Do not generate any additional code, text or example usage information. Make sure the indentation is correct as python is indentation sensitive.
            Do not generate duplicate code.
            """
            code = self.generate_text(prompt)
            if show_planning:
                print(f"Step: {steps_cnt}, plan: {step}")
                steps_cnt += 1
                print(code)
                print("-"*100)
        prompt = f"""
                Given this code: {code}, only output the core code. Remove all the unnecessary information related to examples usages, test cases, etc. If 
                the code does not have extra information, output the code as it is. Output only the required code. Do not output any additional information or additional text.
        """
        code = self.generate_text(prompt)
        return self.clean_code(code, [])
    def generate_code_with_planning_retrieval(self, query, show_planning, k):
        plan = self.generate_plan(query)
        steps = plan.split("||")
        code = ""
        steps_cnt = 1
        for step in steps:
            top_k = self.retrieve_top_k(step, 1)
            ref_codes = [self.sent_to_code[sent] for sent, _ in top_k]
            prompt = f"""
            You are generating code based on a plan. Currently, you have to generate code for the following step: {step}.
            Until now, this code has been generated: {code}. Use the code generated until now to generate the code for the current step.
            Output the final code based on the previous code and by generating code for the current step. You may use this code as reference: {ref_codes}.
            Output the final code upto the current step only. Do not output any additional information or additional text.
            Output only the required code. Do not generate any additional code, text or example usage information. Make sure the indentation is correct as python is indentation sensitive.
            Do not generate duplicate code.
            """
            code = self.generate_text(prompt)
            if show_planning:
                print(f"Step: {steps_cnt}, plan: {step}")
                steps_cnt += 1
                print(code)
                print("-"*100)
        prompt = f"""
                Given this code: {code} and query: {query}, only output the core code. Remove any irrelevant code not relevant to the provided query. Remove all the unnecessary information related to examples usages, test cases, etc. If 
                the code does not have extra information, output the code as it is. Output only the required code. Do not output any additional information or additional text.
                Output only the final required code.
        """
        code = self.generate_text(prompt)
        return self.clean_code(code, [])
    


