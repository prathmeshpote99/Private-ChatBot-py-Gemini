import os
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set. Please check your environment variables.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

def create_chatbot(vector_store):
    """Initialize the chatbot with Google Gemini and FAISS retrieval."""
    try:
        print("🔹 Initializing chatbot with Gemini...")

        model = genai.GenerativeModel("gemini-1.5-pro")
        retriever = vector_store.as_retriever()

        def chatbot_function(query):
            """Retrieves relevant context from FAISS and queries Gemini."""
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"Use the following context:\n{context}\n\nQ: {query}\nA:"
            response = model.generate_content(prompt)
            
            return response.text.strip()

        print("✅ Chatbot Ready with Gemini!")
        return chatbot_function

    except Exception as e:
        print(f"❌ Error creating chatbot: {e}")
        return None

def qa_chatbot(chatbot, query):
    """Query the chatbot."""
    if chatbot is None:
        return "Chatbot is not initialized."

    return chatbot(query)
