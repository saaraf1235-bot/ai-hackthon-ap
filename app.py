import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os, tempfile

st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("DocuMind AI - Chat With Any PDF")

st.markdown("""<style>.stButton>button { background-color: #6C63FF; color: white; border-radius: 10px; }</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    GROQ_API_KEY = st.text_input("Enter your Groq API Key", type="password")
    st.markdown("[Get Free Groq Key](https://console.groq.com/keys)")

if not GROQ_API_KEY:
    st.warning("Please enter Groq API Key to start")
    st.stop()

# No cache - direct
llm = ChatGroq(groq_api_key=GROQ_API_KEY.strip(), model="llama-3.3-70b-versatile")

def ai_logic(query, context):
    prompt = f"""Use context to answer. If not found say "Answer not found in document".
    Context: {context}
    Question: {query}
    Answer:"""
    response = llm.invoke(prompt)
    return response.content

tab1, tab2 = st.tabs(["📄 Document Q&A", "💬 General Chat"])

with tab1:
    st.header("Upload PDF and Ask Questions")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    question = st.text_input("Ask question from PDF")
    if st.button("Get Answer from PDF") and uploaded_file and question:
        with st.spinner("AI is reading PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getbuffer())
                pdf_path = tmp.name
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            context = " ".join([d.page_content for d in chunks[:10]])
            answer = ai_logic(question, context)
            st.success(answer)
            os.remove(pdf_path)

with tab2:
    st.header("General AI Chat")
    user_query = st.text_input("Ask anything", key="gen")
    if st.button("Ask AI") and user_query:
        with st.spinner("AI is thinking..."):
            st.success(ai_logic(user_query, "No context - general knowledge"))
    
   
   
            
