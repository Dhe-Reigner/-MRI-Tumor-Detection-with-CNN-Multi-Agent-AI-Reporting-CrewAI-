from langchain.embeddings import JinaEmbeddings
from langchain.vectorstores import FAISS
from rag.loader import load_medical_docs

def build_vectorstore():
    docs = load_medical_docs()

    embeddings = JinaEmbeddings(
        model_name='jina-embeddings-v4',
        )

    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore