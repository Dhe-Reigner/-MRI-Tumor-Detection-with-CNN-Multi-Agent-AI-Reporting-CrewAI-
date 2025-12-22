from langchain.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
import os


def load_medical_docs(folder_path = 'data'):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path,file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100
    )

    return splitter.split_documents(documents)