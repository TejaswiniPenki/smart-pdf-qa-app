import streamlit as st
import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers import ParentDocumentRetriever

from langgraph.graph import StateGraph, END

# Set API key
genai_api_key = st.text_input("Enter your Google Generative AI API Key", type="password")
if not genai_api_key:
    st.stop()

os.environ["GOOGLE_API_KEY"] = genai_api_key

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if not uploaded_file:
    st.stop()

# Save uploaded file
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_pdf_path = tmp_file.name

# Try different loaders
def load_pdf(path):
    try:
        return PyMuPDFLoader(path).load()
    except:
        try:
            return PyPDFLoader(path).load()
        except:
            try:
                return UnstructuredPDFLoader(path).load()
            except:
                return []

docs = load_pdf(tmp_pdf_path)
if not docs:
    st.error("Failed to load PDF content.")
    st.stop()

# Embeddings and splitting
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
vectorstore = FAISS.from_documents(text_splitter.split_documents(docs), embedding)
retriever = vectorstore.as_retriever()

# QA chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro")

qa_chain = load_qa_chain(llm, chain_type="stuff")

# LangGraph
def retrieve(state):
    query = state["question"]
    docs = retriever.get_relevant_documents(query)
    return {"documents": docs, "question": query}

def generate(state):
    docs = state["documents"]
    query = state["question"]
    if not docs:
        return {"answer": "Answer is not available in the context."}
    result = qa_chain.run(input_documents=docs, question=query)
    return {"answer": result}

builder = StateGraph()
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# QA interface
question = st.text_input("Ask a question about the PDF")
if question:
    output = graph.invoke({"question": question})
    st.success(output["answer"])
