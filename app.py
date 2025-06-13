from flask import Flask, request, jsonify
import os
from pdf_processor import extract_text_from_pdf
from vector_store import create_vector_store
from chatbot import create_chatbot, qa_chatbot
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is not set. Check your .env file or environment variables.")

# Initialize chatbot
chatbot = None

def initialize_chatbot():
    """Initialize chatbot when the app starts."""
    global chatbot
    try:
        print("🔹 Initializing chatbot with Gemini...")
        vector_store = create_vector_store("data/")
        chatbot = create_chatbot(vector_store)
        print("✅ Chatbot Ready with Gemini!")
    except Exception as e:
        print("❌ Error initializing chatbot:", str(e))

# Call chatbot initialization inside app context
with app.app_context():
    initialize_chatbot()

@app.route("/ask", methods=["POST"])
def ask_chatbot():
    """API route to handle chatbot queries."""
    global chatbot
    if chatbot is None:
        return jsonify({"error": "Chatbot not initialized"}), 500
    
    user_input = request.json.get("query", "")
    response = qa_chatbot(chatbot, user_input)
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
