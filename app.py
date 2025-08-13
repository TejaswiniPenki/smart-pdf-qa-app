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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS  # updated as per deprecation warning
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
    uploaded_file.seek(0)  # Reset pointer before reading
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
    uploaded_file.seek(0)  # Reset pointer before reading
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
    uploaded_file.seek(0)  # Reset pointer before reading
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


# ---------- LangGraph Schema + Nodes ----------
# Removed 'answer' to prevent duplicate state key error
class GraphState(TypedDict):
    question: str
    docs: list

def node_semantic(state: GraphState, db):
    docs = db.similarity_search(state["question"], k=6)
    return {"question": state["question"], "docs": docs}

def node_answer(state: GraphState, model, prompt):
    # The state does not hold 'answer', we compute later
    return {}

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("semantic", node_semantic)
    builder.add_node("answer", node_answer)
    builder.add_edge(START, "semantic")
    builder.add_edge("semantic", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


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
    state_data = st.session_state.doc_state

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

    init_state = {"question": adjusted_query, "docs": []}
    graph.invoke(init_state, {"db": state_data["faiss_db"], "model": model, "prompt": prompt})

    # Compute answer after graph execution
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    docs = state_data["faiss_db"].similarity_search(adjusted_query, k=6)
    if is_table_question:
        table_docs = [d for d in docs if "," in d.page_content and "\n" in d.page_content]
        docs = table_docs + [d for d in docs if d not in table_docs]
    answer = chain({"input_documents": docs, "question": question}, return_only_outputs=True).get("output_text", "")

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
