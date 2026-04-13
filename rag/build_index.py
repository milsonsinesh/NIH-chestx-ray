# rag/build_index.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

texts = open("knowledge_base/disease_definitions.txt").read()

splitter = RecursiveCharacterTextSplitter(chunk_size=300)
docs = splitter.split_text(texts)

db = FAISS.from_texts(docs, OpenAIEmbeddings())
db.save_local("rag_index")
