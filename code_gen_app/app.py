import time
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

import torch
import json
import gradio as gr

class CodeGen:
    def __init__(self, vector_db_path, sent_to_code_path, model):
        self.global_calls = 0
        self.api_key = "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"
        self.model = "llama3-70b-8192"
        self.sent_to_code_path = sent_to_code_path
        self.vector_db = torch.load(vector_db_path)
        self.sent_to_code = self.get_sent_to_code()
        # Initialize client for chat model (replace with your actual client setup)
        self.client = Groq(api_key=self.api_key)
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
    def generate_text(self, prompt):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.0,
            max_tokens=800
        )
        self.global_calls += 1
        return chat_completion.choices[0].message.content

    def generate_plan(self, query):
        prompt = f"""
        Given this query: {query}, generate a plan which outlines the steps to solve the problem. This plan should be in such a
        format that generating code for each step of the plan should generate the final code for the query. Output the plan with each
        step separated by a "||". Output only the plan. Do not output any additional information or additional text. In the plan, do not
        include any steps to test the generated code. Your plan should only include the steps to generate the code, modularly.
        """
        return self.generate_text(prompt)

    def generate_code_with_planning(self, query):
        plan = self.generate_plan(query)
        steps = plan.split("||")
        code = ""
        steps_cnt = 1
        for step in steps:
            prompt = f"""
            You are generating code based on a plan. Currently, you have to generate code for the following step: {step}.
            Until now, this code has been generated: {code}. Use the code generated until now to generate the code for the current step.
            Output the final code based on the previous code and by generating code for the current step.
            Output the final code up to the current step only. Do not output any additional information or additional text.
            """
            code = self.generate_text(prompt)
            # Yield the current plan and generated code
            yield f"Step: {steps_cnt}, Plan: {step}\nGenerated Code:\n{code}\n{'-' * 100}"
            time.sleep(1)
            steps_cnt += 1
        yield f"Final Code:\n{self.clean_code(code, [])}"
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
            print("Retrieved Codes: ", codes)
            prompt = f"""
            Generate code for the following query: {query}.
            Make use of these codes as reference while generating the code {codes}. Use the above codes only as reference. Do not copy the code as it is.
            Only output the generated code. Do not output any additional information or additional text.
            Output only the required code. Do not generate any additional code, text or example usage information.
            """
        gen_code = self.generate_text(prompt)
        cleaned_code = self.clean_code(gen_code, codes)
        return cleaned_code
    def generate_code_with_planning_retrieval(self, query, k):
        plan = self.generate_plan(query)
        ref_codes = self.retrieve_top_k(query, k)
        steps = plan.split("||")
        code = ""
        steps_cnt = 1
        print("Retrieved Codes: ", ref_codes)
        for step in steps:
            # Simulated retrieval for this example  # Replace with actual retrieval logic
            prompt = f"""
            You are generating code based on a plan. Currently, you have to generate code for the following step: {step}.
            Until now, this code has been generated: {code}. Use the code generated until now to generate the code for the current step.
            You may use this code as reference: {ref_codes}.
            Output the final code based on the previous code and by generating code for the current step.
            Output the final code up to the current step only. Do not output any additional information or additional text.
            """
            code = self.generate_text(prompt)
            # Yield the current plan and generated code
            # sleep for 1 sec
            time.sleep(1)
            yield f"Step: {steps_cnt}, Plan: {step}\nGenerated Code:\n{code}\n{'-' * 100}"
            steps_cnt += 1
        yield f"Final Code:\n{self.clean_code(code, [])}"

# Initialize CodeGen instance
code_generator = CodeGen("data/vector_db.pt", "data/gold_codes_difficult.json", model="llama3-7b")

# Gradio interface
def generate_code(user_input, retrieval, planning):
    if planning:
        if retrieval:
            stream = code_generator.generate_code_with_planning_retrieval(user_input, 1)
        else:
            stream = code_generator.generate_code_with_planning(user_input)
        for update in stream:
            yield update, None  # Update "Planning Text" box
    else:
        if retrieval:
            code = code_generator.generate_code_retrieve(user_input, 1)
        else:
            code = code_generator.generate_code_retrieve(user_input, 0)
        yield None, code  # Update "Generated Output Code" box
def generate_code_real_time(user_input, retrieval, planning):
    if planning:
        if retrieval:
            stream = code_generator.generate_code_with_planning_retrieval(user_input, 1)
        else:
            stream = code_generator.generate_code_with_planning(user_input)
        # Iterate through the updates for real-time display
        for update in stream:
            yield update, None  # Update "Planning Text" box
    else:
        if retrieval:
            code = code_generator.generate_code_retrieve(user_input, 1)
        else:
            code = code_generator.generate_code_retrieve(user_input, 0)
        yield None, code  # Update "Generated Output Code" box
with gr.Blocks() as demo:
    # Title and description
    gr.Markdown("<h1 style='text-align: center; background-color: #34b9e8; color: white'>Code Generation - Team String</h1>")
    gr.Markdown("An interface to generate code based on user input. Allows for various parameters to be set to customize the generation.")

    with gr.Row():
        retrieval = gr.Checkbox(label="Enable Retrieval", value=False)
        planning = gr.Checkbox(label="Utilize Planning", value=False)

    with gr.Row():
        input_text = gr.Textbox(label="Enter your prompt", lines=5)
        output_text_1 = gr.Code(label="Live Code Updates", language="python", lines=10)
        output_text_2 = gr.Code(label="Generated Output Code", language="python", lines=8)

    submit_btn = gr.Button("Generate", variant="primary", size="lg")

    # Link function to Gradio components
    submit_btn.click(
    fn=generate_code_real_time,
    inputs=[input_text, retrieval, planning],
    outputs=[output_text_1, output_text_2]
)
# Launch the app
demo.launch(share=True)
