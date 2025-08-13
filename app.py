import streamlit as st
import os
import tempfile
import pdfplumber
import pandas as pd
import re
from sklearn.cluster import AgglomerativeClustering

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langgraph.graph import StateGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

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

# ---------- Auto Loader Selection ----------
def auto_select_loader_splitter(uploaded_file):
    uploaded_file.seek(0)  # ✅ Fix: reset pointer before reading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    tables_found = False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page.extract_tables():
                    tables_found = True
                    break
    except Exception:
        tables_found = False

    try:
        docs_test = PyMuPDFLoader(pdf_path).load()
        text_found = any(d.page_content.strip() for d in docs_test)
    except Exception:
        text_found = False

    loader = "UnstructuredPDFLoader" if not text_found else "PyMuPDFLoader"

    os.unlink(pdf_path)
    return loader, "Recursive", tables_found

# ---------- Helpers ----------
def load_pdf_document(uploaded_file, loader_choice):
    uploaded_file.seek(0)  # ✅ Fix: reset pointer before reading
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)

def semantic_chunking(chunks, embedder):
    vectors = embedder.embed_documents(chunks)
    if len(chunks) < 8:
        return list(range(len(chunks)))
    return AgglomerativeClustering(
        n_clusters=min(len(chunks)//5, 12)
    ).fit(vectors).labels_

def extract_tables_as_text(uploaded_file):
    uploaded_file.seek(0)  # 🔹 Ensure pointer reset before reading tables
    table_texts = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                try:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    table_texts.append(df.to_csv(index=False))
                except:
                    pass
    os.unlink(path)
    return table_texts

def create_vector_db(text_chunks, embedder):
    return FAISS.from_texts(text_chunks, embedding=embedder)

# ---------- Question Classification ----------
def classify_question(question):
    table_keywords = [
        "table", "data", "list", "percentage", "amount", "year",
        "GDP", "population", "rate", "value", "total", "figure", "count"
    ]
    if re.search(r"\b\d{4}\b", question) or "%" in question:
        return True
    return any(kw.lower() in question.lower() for kw in table_keywords)

# ---------- Graph Nodes ----------
def node_semantic(q, db):
    return db.similarity_search(q, k=6)

def node_answer(docs, q, model, prompt):
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)(
        {"input_documents": docs, "question": q},
        return_only_outputs=True
    )

def build_graph():
    g = StateGraph()
    g.add_node("semantic", node_semantic)
    g.add_node("answer", node_answer)
    return g

# ---------- Process PDF Once ----------
if pdf_file and api_key and (pdf_file.name != st.session_state.doc_state.get("file_name")):
    st.session_state.doc_state.clear()
    with st.spinner("🔍 Processing PDF..."):
        os.environ["GOOGLE_API_KEY"] = api_key

        loader_choice, _, has_tables = auto_select_loader_splitter(pdf_file)
        docs = load_pdf_document(pdf_file, loader_choice)
        chunks = split_text(docs)

        table_texts = extract_tables_as_text(pdf_file) if has_tables else []
        all_chunks = chunks + table_texts

        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        clusters = semantic_chunking(all_chunks, embedder)
        faiss_db = create_vector_db(all_chunks, embedder)

        st.session_state.doc_state = {
            "file_name": pdf_file.name,
            "faiss_db": faiss_db,
            "embedder": embedder,
            "chunks": all_chunks,
            "clusters": clusters
        }

# ---------- QA ----------
if question and api_key and st.session_state.doc_state:
    state = st.session_state.doc_state

    prompt_template = (
        "Answer using ONLY the provided context.\n"
        "If not found, reply 'Answer is not available in the context.'\n\n"
        "Context:\n{context}\nQuestion:\n{question}\n\nAnswer:"
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    graph = build_graph()

    is_table_question = classify_question(question)
    adjusted_query = "[TABLE PRIORITY] " + question if is_table_question else question

    docs = graph.run("semantic", adjusted_query, state["faiss_db"])

    if is_table_question:
        table_docs = [d for d in docs if "," in d.page_content and "\n" in d.page_content]
        docs = table_docs + [d for d in docs if d not in table_docs]

    response = graph.run("answer", docs, question, model, prompt)
    answer = response.get("output_text", "").strip()

    st.subheader("Answer")
    st.write(answer)

    if show_context:
        st.subheader("Retrieved Context")
        st.write(docs)

elif not pdf_file:
    st.warning("📂 Please upload a PDF.")
elif not api_key:
    st.warning("🔑 Please enter your Google API key.")

st.markdown("---\n✅ Powered by LangGraph, LangChain, FAISS, Google Generative AI")
