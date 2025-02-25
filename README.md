# 🚀 Streamlit-Based RAG Chatbot Assistant  

A **Streamlit-based chatbot assistant** designed for **RAG (Retrieval-Augmented Generation)**, providing precise and contextually relevant answers using **Ollama, LangChain, FAISS, and PyPDF**.  

---

## 📦 Installation & Setup  

### ✅ 1. Make a directory and create a Python Virtual Environment  
Ensure you have Python **3.10** installed, then run:  

For **Ubuntu/Linux/macOS**:  
```bash
mkdir rag_git
cd rag_git
python3.10 -m venv .env_rag_workshop
source .env_rag_workshop/bin/activate
```

### ✅ 2. Clone the Git Repository

```bash
git clone https://github.com/Asrix-AI/llm_rag_workshop.git
cd llm_rag_workshop
```

### ✅ 3. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### ✅ 4. Get API key from langsmith
1. Create an account if you dont have 
2. Sign-in to your account
3. Under Settings, create an API key, save it in your .env file
    Create a `.env` file in the root directory of the project and add the following environment variables:

```bash
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT="https://API.smith.langchain.com"
LANGCHAIN_TRACING_V2=true
```

### ✅ 5. Running Streamlit application
```bash
streamlit run app.py
```
🔹 The application should now be running on http://localhost:8501.
