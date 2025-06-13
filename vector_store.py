import faiss
import pickle
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(data_dir):
    """Load or build FAISS vector database."""
    index_file = os.path.join(data_dir, "faiss_index.pkl")

    if os.path.exists(index_file):
        print("🔹 Loading existing FAISS index...")
        with open(index_file, "rb") as f:
            vector_store = pickle.load(f)
    else:
        print("⚠️ No FAISS index found. Creating a new one...")
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(["Hello, I am a chatbot!"], embeddings)

        # Save the index
        with open(index_file, "wb") as f:
            pickle.dump(vector_store, f)

        print("✅ FAISS index created successfully!")

    return vector_store
