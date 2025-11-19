# SCRPIT PARA GERAR A BASE DE DADOS
# todos PDFs devem ficar na pasta /rag/data

#BIBLIOTECAS
import os

from decouple import config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# chama as APIs do Groq e do Hugging Face

os.environ["API_KEY_HERE"] = config("API_KEY_HERE")
os.environ["API_KEY_HERE"] = config("API_KEY_HERE")

# Carrega o PDF e transforma em texto
if __name__ == '__main__':
    file_path = '/app/rag/data/django_master.pdf'
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

# Divide o texto em chunks 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(
        documents=docs,
    )

# Cria o banco vetorial - Chroma
    persist_directory = '/app/chroma_data'

    # Gera embeddings com o Hugging Face
    embedding = HuggingFaceEmbeddings()
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
    )

    # Por fim, salva tudo no banco vetorial
    vector_store.add_documents(
        documents=chunks,
    )