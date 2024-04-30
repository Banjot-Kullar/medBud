from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json
import streamlit as st

st.title("Medical RAG")


app = FastAPI()

templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
model_path= local_llm,
temperature=0.1,
max_tokens=2048,
top_p=1
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""
embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
print("Embeddings Initialized....")
url = "http://localhost:6333"

client = QdrantClient(
url=url, prefer_grpc=False
)
print("client set")

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db2")
print("DB set")
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
print("Prompt set")
retriever = db.as_retriever(search_kwargs={"k":1})
print("Retriever set")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

print("Server Started....")
query = st.text_input("Enter your Query")
print(f"Received query: {query}")

chain_type_kwargs = {"prompt": prompt}

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
response = qa(query)
print(f"Generated response: {response}")
answer = response['result']
source_document = response['source_documents'][0].page_content
doc = response['source_documents'][0].metadata['source']
response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))

res = Response(response_data)
print(res)
st.write(answer)
st.write(doc)
#st.write(source_document)

# @app.post("/get_response")
# async def get_response(query: str = Form(...)):
#     print(f"Received query: {query}")
#     chain_type_kwargs = {"prompt": prompt}
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
#     response = qa(query)
#     print(f"Generated response: {response}")
#     answer = response['result']
#     source_document = response['source_documents'][0].page_content
#     doc = response['source_documents'][0].metadata['source']
#     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
#     res = Response(response_data)
#     print(res)
#     return res