from typing import Union
from fastapi import FastAPI
from decouple import config
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

app = FastAPI()
openai.api_key = config('OPENAI_API_KEY')

@app.get("/")
def read_root():
    return {"welcome to de API GalileiLerning"}


@app.get("/openai/{send_message}/{send_resum}")
def read_item(send_message: str, send_resum: str):

    response= openai.Embedding.create(
      engine="text-similarity-davinci-001",
      input=[send_message, send_resum])
    embedding_text=response['data'][0]['embedding']
    embedding_answer=response['data'][1]['embedding']

    np.dot(embedding_text,embedding_answer)
    response_message = np.dot(embedding_text,embedding_answer)
    return {"response_message": response_message}