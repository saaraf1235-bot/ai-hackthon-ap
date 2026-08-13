import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

st.set_page_config(page_title="AI Hackathon App", layout="wide")
st.title("🚀 AI Hackathon Master App - No Torch Version")

GROQ_API_KEY = st.text_input("Enter your Groq API Key", type="password")

@st.cache_resource
def setup_ai(_api_key):
    llm = ChatGroq(groq_api_key=_api_key, model_name="llama-3.1-8b-instant")
    return llm

def ai_logic(query, context, llm):
    prompt = f"""You are a helpful AI assistant.
    Use the following context to answer the question.
    If the answer is not in context, say "Answer not found in document".
    
    Context: {context}
    
    Question: {query}
    Answer:"""
    response = llm.invoke(prompt)
    return response.content

if GROQ_API_KEY:
    llm = setup_ai(GROQ_API_KEY)
else:
    st.warning("Please enter Groq API Key to start")
    st.stop()

tab1, tab2 = st.tabs(["📄 Document Q&A", "💬 General Chat"])

with tab1:
    st.header("Upload PDF and Ask Questions")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    question = st.text_input("Ask question from PDF")
    
    if st.button("Get Answer from PDF") and uploaded_file and question:
        with st.spinner("AI is reading PDF and thinking..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                pdf_path = tmp.name
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            context = " ".join([d.page_content for d in chunks[:10]]) # First 10 chunks only
            
            answer = ai_logic(question, context, llm)
            st.success(answer)
            os.remove(pdf_path)

with tab2:
    st.header("General AI Chat")
    user_query = st.text_input("Ask anything")
    if st.button("Ask AI") and user_query:
        with st.spinner("AI is thinking..."):
            answer = ai_logic(user_query, "No context", llm)
            st.success(answer)
