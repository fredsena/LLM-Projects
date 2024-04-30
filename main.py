from typing import Union
from fastapi import FastAPI
from fastapi import FastAPI, Request
# from transformers import pipeline
from pydantic import BaseModel

import os
# from langchain.llms import CTransformers
# from langchain import PromptTemplate, LLMChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from huggingface_hub import hf_hub_download
# from langchain import PromptTemplate, HuggingFaceHub, LLMChain

app = FastAPI()

# Create a class for the input data
class InputData(BaseModel):
 prompt: str

# Create a class for the output data
class OutputData(BaseModel):
 response: str

# Load a local LLM using Hugging Face Transformers
# You can change the model name and the task according to your needs
# For example, you can use “t5-base” for summarization or “bert-base-cased” for question answering
# model = pipeline("text-generation", model="gpt2")
# classifier = pipeline('sentiment-analysis')

# GENERATIVE_AI_MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
# GENERATIVE_AI_MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# model_path = hf_hub_download(
#     repo_id=GENERATIVE_AI_MODEL_REPO,
#     filename=GENERATIVE_AI_MODEL_FILE
# )

# print(model_path)

config = {
    "max_new_tokens": 2048,
    "context_length": 4096,
    "repetition_penalty": 1.1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 0.9,
    "stream": True,
    "threads": int(os.cpu_count() / 2)
}

# llm = CTransformers(model=model_path, config=config)
# template = """Question: {question}

# Answer: Let's think step by step."""

# promptBase = PromptTemplate(template=template, input_variables=["question"])
# llm_chain = LLMChain(prompt=promptBase, llm=llm)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# # Create a route for the web application
# @app.post("/generate", response_model=OutputData)
# async def generate(request: Request, input_data: InputData):
#  # Get the prompt from the input data
#  prompt = input_data.prompt

# # Generate a response from the local LLM using the prompt
#  response = model(prompt)[0]["generated_text"]

# # Return the response as output data
#  return OutputData(response=response)


# @app.post("/sentiment-analysis", response_model=OutputData)
# async def sentiment(request: Request, input_data: InputData): 
#  prompt = input_data.prompt
#  response = classifier(prompt)
#  return OutputData(response=str(response))

# local LLM Mistral-7B
# @app.post("/mistral7B", response_model=OutputData)
# async def mistral7B(request: Request, input_data: InputData):
#  prompt = input_data.prompt
#  response = llm_chain.run(prompt)
#  return OutputData(response=str(response))