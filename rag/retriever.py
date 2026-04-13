# rag/retriever.py

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

db = FAISS.load_local("rag_index", OpenAIEmbeddings())

def retrieve(query):
    return db.similarity_search(query, k=3)
