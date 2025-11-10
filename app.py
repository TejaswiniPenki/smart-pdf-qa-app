import streamlit as st
import os
import tempfile
import pdfplumber
import pandas as pd
import re
from sklearn.cluster import AgglomerativeClustering
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="LangGraph PDF QA", page_icon="📄", layout="wide")
st.title("📄 LangGraph PDF QA")
st.markdown("Upload a PDF and ask questions — answering from both theory text and tables.")

api_key = st.sidebar.text_input("Google API key:", type="password")
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
question = st.sidebar.text_area("Ask a question:")
show_context = st.sidebar.checkbox("Show retrieved context for debugging")
if "doc_state" not in st.session_state:
    st.session_state.doc_state = {}

# ---------- Auto Loader ----------
def auto_select_loader_splitter(uploaded_file):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
    try:
        docs_test = PyMuPDFLoader(pdf_path).load()
        text_found = any(d.page_content.strip() for d in docs_test)
    except Exception:
        text_found = False
    loader = "UnstructuredPDFLoader" if not text_found else "PyMuPDFLoader"
    os.unlink(pdf_path)
    return loader, "Recursive", True

# ---------- Helpers ----------
def load_pdf_document(uploaded_file, loader_choice):
    uploaded_file.seek(0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    if loader_choice == "PyPDFLoader":
        docs = PyPDFLoader(path).load()
    elif loader_choice == "PyMuPDFLoader":
        docs = PyMuPDFLoader(path).load()
    else:
        docs = UnstructuredPDFLoader(path).load()
    os.unlink(path)
    return docs

def split_text(documents):
    text = " ".join([doc.page_content for doc in documents])
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_text(text)

def semantic_chunking(chunks, embedder):
    vectors = embedder.embed_documents(chunks)
    if len(chunks) < 8:
        return list(range(len(chunks)))
    return AgglomerativeClustering(n_clusters=min(len(chunks)//5, 12)).fit(vectors).labels_

# ---------- Robust Table Extraction ----------
def extract_tables_as_text(uploaded_file):
    uploaded_file.seek(0)
    table_texts, table_dfs, table_count = [], [], 0
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_count += len(tables)
            for t in tables:
                try:
                    if not t or len(t) < 2:
                        continue
                    headers = [
                        str(h).strip() if h and str(h).strip() != "" else f"Column_{i+1}"
                        for i, h in enumerate(t[0])
                    ]
                    seen = {}
                    for i, h in enumerate(headers):
                        if h in seen:
                            seen[h] += 1
                            headers[i] = f"{h}_{seen[h]}"
                        else:
                            seen[h] = 1
                    df = pd.DataFrame(t[1:], columns=headers)
                    df = df.dropna(how="all").reset_index(drop=True)
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    table_dfs.append(df)
                    table_texts.append(df.to_markdown(index=False))
                except Exception:
                    pass
    os.unlink(path)
    return table_texts, table_dfs, table_count

def create_vector_db(text_chunks, embedder):
    return FAISS.from_texts(text_chunks, embedding=embedder)

# ---------- LangGraph ----------
class GraphState(TypedDict):
    question: str
    docs: list

def node_semantic(state: GraphState, **kwargs):
    db = kwargs.get("db")
    docs = db.similarity_search(state["question"], k=6) if db else []
    return {"question": state["question"], "docs": docs}

def node_answer(state: GraphState, **kwargs):
    model = kwargs.get("model")
    docs = state["docs"]
    if model and docs:
        qa_chain = create_stuff_documents_chain(model)
        _ = qa_chain.invoke({"input_documents": docs, "question": state["question"]})
    return {}

def build_graph():
    b = StateGraph(GraphState)
    b.add_node("semantic", node_semantic)
    b.add_node("answer", node_answer)
    b.add_edge(START, "semantic")
    b.add_edge("semantic", "answer")
    b.add_edge("answer", END)
    return b.compile()

# ---------- Process PDF ----------
if pdf_file and api_key and (pdf_file.name != st.session_state.doc_state.get("file_name")):
    st.session_state.doc_state.clear()
    with st.spinner("🔍 Processing PDF..."):
        os.environ["GOOGLE_API_KEY"] = api_key
        loader_choice, _, _ = auto_select_loader_splitter(pdf_file)
        docs = load_pdf_document(pdf_file, loader_choice)
        chunks = split_text(docs)
        table_texts, table_dfs, table_count = extract_tables_as_text(pdf_file)
        all_chunks = chunks + table_texts
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        clusters = semantic_chunking(all_chunks, embedder)
        faiss_db = create_vector_db(all_chunks, embedder)
        st.session_state.doc_state = {
            "file_name": pdf_file.name,
            "faiss_db": faiss_db,
            "embedder": embedder,
            "chunks": all_chunks,
            "clusters": clusters,
            "table_count": table_count,
            "table_dfs": table_dfs
        }

# ---------- QA ----------
if question and api_key and st.session_state.doc_state:
    data = st.session_state.doc_state
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, convert_system_message_to_human=True)
    graph = build_graph()
    adj_q = question
    init_state = {"question": adj_q, "docs": []}
    graph.invoke(init_state, {"db": data["faiss_db"], "model": model})
    docs = data["faiss_db"].similarity_search(adj_q, k=6)
    qa_chain = create_stuff_documents_chain(model)
    ans = qa_chain.invoke({"input_documents": docs, "question": question})
    st.subheader("Answer")
    st.write(ans)
    if show_context:
        st.subheader("Retrieved Context")
        st.write(docs)
elif not pdf_file:
    st.warning("📂 Please upload a PDF.")
elif not api_key:
    st.warning("🔑 Please enter your Google API key.")
