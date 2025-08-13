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
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
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


# ---------- Auto Loader Selection ----------
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
    return loader, "Recursive", True  # force table extraction


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
                    df = pd.DataFrame(t[1:], columns=t[0])
                    table_dfs.append(df)
                    table_texts.append(df.to_csv(index=False))
                except:
                    pass
    os.unlink(path)
    return table_texts, table_dfs, table_count

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


# ---------- Special Query Detection ----------
def is_table_count_question(q):
    ql = q.lower()
    return any(k in ql for k in [
        "how many tables", "number of tables", "tables extracted",
        "total tables", "count of tables", "tables in pdf", "tables found"
    ])
def is_row_request(q): return bool(re.search(r"row\s+\d+", q.lower()))
def is_column_request(q): return bool(re.search(r"column\s+[a-zA-Z0-9_ ]+", q.lower()))
def is_pdf_summary_request(q):
    ql = q.lower()
    return "summarise pdf" in ql or "summary of pdf" in ql or "pdf summary" in ql
def is_table_summary_request(q):
    return bool(re.search(r"(summarise|summary of)\s+table\s+\d+", q.lower()))
# keyword table search
def is_table_search_request(q):
    return re.search(r"(find|show)\s+rows?\s+where\s+", q.lower()) is not None
def parse_table_search_request(q):
    table_idx = 0
    tm = re.search(r"table\s+(\d+)", q.lower())
    if tm:
        table_idx = int(tm.group(1)) - 1
    m = re.search(r"where\s+([a-zA-Z0-9_ ]+)\s*=\s*([a-zA-Z0-9_ ]+)", q.lower())
    if m:
        return table_idx, m.group(1).strip(), m.group(2).strip()
    return None, None, None


# ---------- LangGraph Schema + Nodes ----------
class GraphState(TypedDict):
    question: str
    docs: list

def node_semantic(state: GraphState, **kwargs):
    db = kwargs.get("db")
    docs = db.similarity_search(state["question"], k=6) if db else []
    return {"question": state["question"], "docs": docs}

def node_answer(state: GraphState, **kwargs):
    model = kwargs.get("model"); prompt = kwargs.get("prompt")
    if model and prompt:
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        _ = chain({"input_documents": state["docs"], "question": state["question"]},
                  return_only_outputs=True)
    return {}

def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("semantic", node_semantic)
    builder.add_node("answer", node_answer)
    builder.add_edge(START, "semantic")
    builder.add_edge("semantic", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


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
    state_data = st.session_state.doc_state
    # special cases
    if is_table_count_question(question):
        c = state_data.get("table_count", 0)
        st.subheader("Answer"); st.write(f"There are {c} tables extracted from the PDF." if c else "No tables detected.")
        st.stop()
    if is_row_request(question) and state_data.get("table_dfs"):
        m = re.search(r"row\s+(\d+)", question.lower()); row_idx = int(m.group(1))-1
        table_idx = 0; tm = re.search(r"table\s+(\d+)", question.lower())
        if tm: table_idx = int(tm.group(1))-1
        try: st.subheader("Answer"); st.write(state_data["table_dfs"][table_idx].iloc[row_idx].to_dict())
        except: st.write("Requested row not found.")
        st.stop()
    if is_column_request(question) and state_data.get("table_dfs"):
        m = re.search(r"column\s+([a-zA-Z0-9_ ]+)", question.lower()); col = m.group(1).strip()
        table_idx = 0; tm = re.search(r"table\s+(\d+)", question.lower())
        if tm: table_idx = int(tm.group(1))-1
        try: st.subheader("Answer"); st.write(state_data["table_dfs"][table_idx][col].tolist())
        except: st.write("Requested column not found.")
        st.stop()
    if is_table_search_request(question) and state_data.get("table_dfs"):
        t_idx, col, val = parse_table_search_request(question)
        if col and val and 0 <= t_idx < len(state_data["table_dfs"]):
            try:
                matches = state_data["table_dfs"][t_idx][state_data["table_dfs"][t_idx][col].astype(str).str.lower() == val.lower()]
                st.subheader(f"Rows in Table {t_idx+1} where {col} = {val}")
                st.write(matches.to_dict(orient="records") if not matches.empty else "No matching rows found.")
            except KeyError:
                st.write(f"Column '{col}' not found.")
        else:
            st.write("Could not parse search query.")
        st.stop()
    if is_pdf_summary_request(question):
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        chain = load_qa_chain(model, chain_type="stuff")
        summary_doc = Document(page_content="\n".join(state_data["chunks"]))
        summary = chain({"input_documents": [summary_doc], "question": "Summarise the PDF"},
                        return_only_outputs=True).get("output_text", "")
        st.subheader("PDF Summary"); st.write(summary); st.stop()
    if is_table_summary_request(question) and state_data.get("table_dfs"):
        tm = re.search(r"table\s+(\d+)", question.lower()); t_idx = int(tm.group(1))-1
        if 0 <= t_idx < len(state_data["table_dfs"]):
            csv_text = state_data["table_dfs"][t_idx].to_csv(index=False)
            model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
            chain = load_qa_chain(model, chain_type="stuff")
            table_doc = Document(page_content=csv_text)
            summary = chain({"input_documents": [table_doc], "question": "Summarise this table"},
                            return_only_outputs=True).get("output_text", "")
            st.subheader(f"Summary of Table {t_idx+1}"); st.write(summary)
        else:
            st.write("Table not found.")
        st.stop()

    # default vector QA
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
    docs = state_data["faiss_db"].similarity_search(adjusted_query, k=6)
    if is_table_question:
        table_docs = [d for d in docs if "," in d.page_content and "\n" in d.page_content]
        docs = table_docs + [d for d in docs if d not in table_docs]
    context_text = "\n".join([d.page_content for d in docs])
    answer = load_qa_chain(model, chain_type="stuff", prompt=prompt)(
        {"context": context_text, "question": question},
        return_only_outputs=True
    ).get("output_text", "")
    st.subheader("Answer"); st.write(answer)
    if show_context:
        st.subheader("Retrieved Context"); st.write(docs)
elif not pdf_file:
    st.warning("📂 Please upload a PDF.")
elif not api_key:
    st.warning("🔑 Please enter your Google API key.")

st.markdown("---\n✅ Powered by LangGraph, LangChain, FAISS, Google Generative AI")
