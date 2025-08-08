# app.py
import os
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

# ------------------- SETUP -------------------
api_key = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else st.text_input("Enter your Google API Key:", type="password")
os.environ["GOOGLE_API_KEY"] = api_key

# ------------------- FUNCTIONS -------------------
def detect_pdf_type(file_path):
    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        text = " ".join([doc.page_content for doc in docs])
        if "table" in text.lower():
            return "table"
        elif len(text) > 10000:
            return "longform"
        return "standard"
    except:
        return "unknown"

def get_loader_by_type(file_path, pdf_type):
    if pdf_type == "table":
        return UnstructuredPDFLoader(file_path)
    elif pdf_type == "longform":
        return PyMuPDFLoader(file_path)
    return PyPDFLoader(file_path)

def get_splitter_by_type(text):
    if len(text) > 15000:
        return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    elif "table" in text.lower():
        return CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)

def create_parent_retriever(docs, splitter):
    child_chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(child_chunks, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    store = InMemoryStore()
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=splitter
    )
    retriever.add_documents(docs)
    return retriever

# ------------------- LANGGRAPH -------------------
@RunnableLambda
def retrieve_chunk(state):
    query, retriever = state["question"], state["retriever"]
    docs = retriever.get_relevant_documents(query)
    return {"docs": docs, **state}

@RunnableLambda
def generate_answer(state):
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use only the following context to answer.

    Context:
    {context}

    Question: {question}
    
    If answer is not present in the context, say 'Answer is not available in the context.'
    """)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    chain = prompt | llm
    docs = state["docs"]
    context = "\n\n".join([doc.page_content for doc in docs])
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"answer": response.content, **state}

builder = StateGraph()
builder.add_node("Retrieve", retrieve_chunk)
builder.add_node("Generate", generate_answer)
builder.set_entry_point("Retrieve")
builder.add_edge("Retrieve", "Generate")
builder.set_finish_point("Generate")
graph = builder.compile()

# ------------------- STREAMLIT UI -------------------
st.title("📄 Smart PDF QA with LangGraph")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    pdf_type = detect_pdf_type(file_path)
    loader = get_loader_by_type(file_path, pdf_type)
    docs = loader.load()
    all_text = " ".join([doc.page_content for doc in docs])
    splitter = get_splitter_by_type(all_text)
    retriever = create_parent_retriever(docs, splitter)

    question = st.text_input("Ask a question about the PDF:")
    if question:
        output = graph.invoke({"question": question, "retriever": retriever})
        st.success(output["answer"])
