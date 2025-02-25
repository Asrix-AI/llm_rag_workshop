import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain_ollama.embeddings import OllamaEmbeddings


# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
db_name = "health_supplements"
vector_store = FAISS.load_local(db_name, embeddings=embeddings, allow_dangerous_deserialization=True)

# Set up retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 1})

# Initialize ChatOllama model
model = ChatOllama(
    model="llama3.2:1b",
    base_url="http://localhost:11434",
    streaming=True,  # Enable streaming
)

# Define the prompt
prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Make sure your answer is relevant to the question and it is answered from the context only. Only answer in English.
    Question: {question} 
    Context: {context} 
    Answer:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="PrivateGPT",
    page_icon="logo.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit app title
st.title("PrivateGPT")
# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    st.image("logo.webp", use_container_width=True)
    st.markdown("### 📚 Your Private Document Assistant")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.markdown("""
    **Your AI-powered assistant for document understanding.**  
     
    ### 🔹 **Built using Open Source Stack:**  
    - 🦙 **Llama 3.2** – Powerful LLM for intelligent responses.  
    - 🏗 **Nomic-Embed-Text** – Converts text into meaningful embeddings.  
    - 🔍 **FAISS** – Enables fast and accurate document retrieval.  
     
    ### 🎯 **What Can PrivateGPT Do?**  
    - **📄 Summarize** – Get concise summaries of documents.  
    - **💬 Chat** – Ask questions and interact with your documents.  
    - **🔎 Retrieve** – Find relevant information instantly.  

    """)

elif choice == "🤖 Chatbot":   

    # Display all previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user query
    if user_input := st.chat_input("Chat with your docs...."):
        # Append user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Format the prompt
        formatted_query = prompt.format_messages(question=user_input, context=context)

        # Stream assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("⌛ Thinking...")  # Show a loading indicator

            response = ""
            response_stream = model.stream([HumanMessage(content=formatted_query[0].content)])
            for chunk in response_stream:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                response += token
                response_placeholder.markdown(response)

        # Append assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})

