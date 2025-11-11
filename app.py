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
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document


# -------- Dynamic Embedding/Model Selection --------
def get_api_choice():
    secrets = st.secrets
    if all(x in secrets for x in ["VERTEX_API_KEY", "PROJECT_ID", "LOCATION"]):
        return "vertex"
    elif "GOOGLE_API_KEY" in secrets:
        return "gemini"
    else:
        return None

def get_embedder(api_choice):
    if api_choice == "vertex":
        # VertexAIEmbeddings requires vertex secrets
        from langchain_google_vertexai import VertexAIEmbeddings
        os.environ["VERTEX_API_KEY"] = st.secrets["VERTEX_API_KEY"]
        os.environ["PROJECT_ID"] = st.secrets["PROJECT_ID"]
        os.environ["LOCATION"] = st.secrets["LOCATION"]
        return VertexAIEmbeddings(model_name="gemini-embedding-001")
    elif api_choice == "gemini":
        # GoogleGenerativeAIEmbeddings for Gemini API
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    else:
        st.error("No valid API key configuration found! Check secrets configuration.")
        st.stop()

def get_chat_model(api_choice):
    from langchain_google_genai import ChatGoogleGenerativeAI
    if api_choice == "vertex":
        # Vertex API: Provide extra options if needed
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=st.secrets["VERTEX_API_KEY"],
            project=st.secrets["PROJECT_ID"],
            location=st.secrets["LOCATION"])
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.3,
            api_key=st.secrets["GOOGLE_API_KEY"])

# ---------- Streamlit Setup ----------
st.set_page_config(page_title="LangGraph PDF QA", page_icon="📄", layout="wide")
st.title("📄 LangGraph PDF QA")
st.markdown("Upload a PDF and ask questions — answering from both theory text and tables.")

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
                    # Clean and deduplicate headers
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
                except Exception as e:
                    print("Table parse error:", e)
                    pass
    os.unlink(path)
    return table_texts, table_dfs, table_count

def create_vector_db(text_chunks, embedder):
    return FAISS.from_texts(text_chunks, embedding=embedder)

# ---------- Question Classification ----------
def classify_question(question):
    kws = ["table", "data", "list", "percentage", "amount", "year", "GDP", "population", "rate", "value", "total", "figure", "count"]
    if re.search(r"\b\d{4}\b", question) or "%" in question:
        return True
    return any(k.lower() in question.lower() for k in kws)

# ---------- Special Query Detection & Parsing ----------
def is_table_count_question(q): 
    return any(k in q.lower() for k in ["how many tables", "number of tables","tables extracted","total tables","count of tables","tables in pdf","tables found"])
def is_row_request(q): 
    return bool(re.search(r"row\s+\d+", q.lower()))
def is_column_request(q): 
    return bool(re.search(r"column\s+[a-zA-Z0-9_ ]+", q.lower()))
def is_pdf_summary_request(q): 
    return any(x in q.lower() for x in ["summarise pdf", "summary of pdf", "pdf summary"])
def is_table_summary_request(q): 
    return bool(re.search(r"(summarise|summary of)\s+table\s+\d+", q.lower()))
def is_table_search_request(q): 
    return re.search(r"(find|show)\s+rows?\s+where\s+", q.lower()) is not None
def is_row_and_column_request(q):
    return bool(re.search(r"row\s+\d+", q.lower()) and re.search(r"column\s+[a-zA-Z0-9_ ]+", q.lower()))
def is_column_where_request(q):
    return bool(re.search(r"(display|show)\s+column\s+[a-zA-Z0-9_ ]+\s+where\s+", q.lower()))

def parse_table_search_request(q):
    table_idx = 0
    tm = re.search(r"table\s+(\d+)", q.lower())
    if tm: 
        table_idx = int(tm.group(1)) - 1
    m_eq = re.search(r"where\s+([a-zA-Z0-9_ ]+)\s*=\s*([a-zA-Z0-9_ ]+)", q.lower())
    m_ct = re.search(r"where\s+([a-zA-Z0-9_ ]+)\s+contains\s+([a-zA-Z0-9_ ]+)", q.lower())
    if m_eq:
        return table_idx, m_eq.group(1).strip(), m_eq.group(2).strip(), "exact"
    if m_ct:
        return table_idx, m_ct.group(1).strip(), m_ct.group(2).strip(), "contains"
    return None, None, None, None

def parse_column_where_request(q):
    table_idx = 0
    tm = re.search(r"table\s+(\d+)", q.lower())
    if tm:
        table_idx = int(tm.group(1)) - 1
    m_eq = re.search(
        r"(?:display|show)\s+column\s+([a-zA-Z0-9_ ]+)\s+where\s+([a-zA-Z0-9_ ]+)\s*=\s*([a-zA-Z0-9_ ]+)",
        q.lower()
    )
    m_ct = re.search(
        r"(?:display|show)\s+column\s+([a-zA-Z0-9_ ]+)\s+where\s+([a-zA-Z0-9_ ]+)\s+contains\s+([a-zA-Z0-9_ ]+)",
        q.lower()
    )
    if m_eq:
        return table_idx, m_eq.group(1).strip(), m_eq.group(2).strip(), m_eq.group(3).strip(), "exact"
    if m_ct:
        return table_idx, m_ct.group(1).strip(), m_ct.group(2).strip(), m_ct.group(3).strip(), "contains"
    return None, None, None, None, None

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
    prompt = kwargs.get("prompt")
    if model and prompt:
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        _ = chain({"input_documents": state["docs"], "question": state["question"]}, return_only_outputs=True)
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
api_choice = get_api_choice()
if not api_choice:
    st.error("No valid API key present in secrets! Please add either GOOGLE_API_KEY or VERTEX_API_KEY with PROJECT_ID/LOCATION.")
    st.stop()

if pdf_file and (pdf_file.name != st.session_state.doc_state.get("file_name")):
    st.session_state.doc_state.clear()
    with st.spinner("🔍 Processing PDF..."):
        loader_choice, _, _ = auto_select_loader_splitter(pdf_file)
        docs = load_pdf_document(pdf_file, loader_choice)
        chunks = split_text(docs)
        table_texts, table_dfs, table_count = extract_tables_as_text(pdf_file)
        all_chunks = chunks + table_texts
        embedder = get_embedder(api_choice)
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
if question and st.session_state.doc_state:
    data = st.session_state.doc_state
    # Row + Column single cell lookup (Pandas for accuracy)
    if is_row_and_column_request(question) and data.get("table_dfs"):
        rm = re.search(r"row\s+(\d+)", question.lower())
        cm = re.search(r"column\s+([a-zA-Z0-9_ ]+)", question.lower())
        if rm and cm:
            row_idx = int(rm.group(1)) - 1
            col = cm.group(1).strip()
            table_idx = 0
            tm = re.search(r"table\s+(\d+)", question.lower())
            if tm:
                table_idx = int(tm.group(1)) - 1
            try:
                value = data["table_dfs"][table_idx].iloc[row_idx][col]
                st.subheader("Answer")
                st.write(f"(Cell Lookup) Value in row {row_idx+1}, column '{col}': {value}")
            except Exception as e:
                st.write("Requested row/column not found.")
        st.stop()

    # Conditional column display (Pandas filter first)
    if is_column_where_request(question) and data.get("table_dfs"):
        t_idx, target_col, filter_col, filter_val, match_type = parse_column_where_request(question)
        if all([target_col, filter_col, filter_val]) and 0 <= t_idx < len(data["table_dfs"]):
            df = data["table_dfs"][t_idx]
            try:
                if match_type == "exact":
                    matches = df[df[filter_col].astype(str).str.lower() == filter_val.lower()]
                elif match_type == "contains":
                    matches = df[df[filter_col].astype(str).str.lower().str.contains(filter_val.lower(), na=False)]
                else:
                    matches = pd.DataFrame()
                if target_col in df.columns:
                    st.subheader(f"{target_col} values where {filter_col} {match_type} '{filter_val}' (Table {t_idx+1})")
                    st.write(matches[target_col].tolist())
                else:
                    st.write(f"Column '{target_col}' not found in Table {t_idx+1}.")
            except KeyError:
                st.write(f"Column '{filter_col}' not found in Table {t_idx+1}.")
        else:
            st.write("Could not parse 'display column where' query properly.")
        st.stop()

    # Existing special cases
    if is_table_count_question(question):
        st.subheader("Answer")
        st.write(f"There are {data['table_count']} tables." if data["table_count"] else "No tables detected.")
        st.stop()

    if is_row_request(question) and data["table_dfs"]:
        m = re.search(r"row\s+(\d+)", question.lower())
        row_idx = int(m.group(1))-1
        tidx = 0
        tm = re.search(r"table\s+(\d+)", question.lower())
        if tm: tidx = int(tm.group(1))-1
        try:
            st.subheader("Answer")
            row_dict = data["table_dfs"][tidx].iloc[row_idx].to_dict()
            st.write(f"(Row Lookup) Row {row_idx+1} in Table {tidx+1}:\n{row_dict}")
        except:
            st.write("Requested row not found.")
        st.stop()

    if is_column_request(question) and data["table_dfs"]:
        m = re.search(r"column\s+([a-zA-Z0-9_ ]+)", question.lower())
        col = m.group(1).strip()
        tidx = 0
        tm = re.search(r"table\s+(\d+)", question.lower())
        if tm: tidx = int(tm.group(1))-1
        try:
            st.subheader("Answer")
            col_list = data["table_dfs"][tidx][col].tolist()
            st.write(f"(Column Lookup) Values in column '{col}' of Table {tidx+1}:")
            st.write(col_list)
        except:
            st.write("Requested column not found.")
        st.stop()

    # Conditional row display/filter (Pandas filter first)
    if is_table_search_request(question) and data["table_dfs"]:
        t_idx, col, val, match_type = parse_table_search_request(question)
        if col and val and 0 <= t_idx < len(data["table_dfs"]):
            df = data["table_dfs"][t_idx]
            try:
                if match_type == "exact":
                    matches = df[df[col].astype(str).str.lower() == val.lower()]
                elif match_type == "contains":
                    matches = df[df[col].astype(str).str.lower().str.contains(val.lower(), na=False)]
                else:
                    matches = pd.DataFrame()
                st.subheader(f"Rows in Table {t_idx+1} where {col} {match_type} '{val}'")
                st.write(matches.to_dict(orient="records") if not matches.empty else "No matching rows found.")
            except KeyError:
                st.write(f"Column '{col}' not found.")
        else:
            st.write("Could not parse query.")
        st.stop()

    # ----------- PDF Summary (robust/factual with Gemini) -----------
    if is_pdf_summary_request(question):
        model = get_chat_model(api_choice)
        # Only use the first 10 chunks for summary
        doc = Document(page_content="\n".join(data["chunks"][:10]))
        summary_prompt = (
            "You are a professional summarizer. "
            "Create a concise summary of the following PDF content, only highlighting real topics and information. "
            "Do not invent any facts. Be factual and brief."
        )
        ans = load_qa_chain(model, chain_type="map_reduce")(
            {"input_documents": [doc], "question": summary_prompt},
            return_only_outputs=True
        )
        st.subheader("PDF Summary")
        st.write(ans.get("output_text", ""))
        st.stop()

    # ----------- Table Summary (robust/factual with Gemini) -----------
    if is_table_summary_request(question) and data["table_dfs"]:
        tm = re.search(r"table\s+(\d+)", question.lower())
        tidx = int(tm.group(1)) - 1
        if 0 <= tidx < len(data["table_dfs"]):
            df = data["table_dfs"][tidx]
            table_markdown = df.to_markdown(index=False)
            doc = Document(page_content=f"Here is a data table in markdown format:\n{table_markdown}")
            summary_prompt = (
                "You are a data analyst. Summarize only facts from the following table. "
                "List the row and column counts, any highest and lowest values, and major trends. "
                "Do NOT make up any numbers or content. Be concise and factual."
            )
            model = get_chat_model(api_choice)
            ans = load_qa_chain(model, chain_type="map_reduce")(
                {"input_documents": [doc], "question": summary_prompt},
                return_only_outputs=True
            )
            st.subheader(f"Summary of Table {tidx+1}")
            st.write(ans.get("output_text", ""))
        else:
            st.write("Table not found.")
        st.stop()

    # ----------- Default Vector QA -----------
    model = get_chat_model(api_choice)
    graph = build_graph()
    is_tbl_q = classify_question(question)
    adj_q = "[TABLE PRIORITY] " + question if is_tbl_q else question
    init_state = {"question": adj_q, "docs": []}
    graph.invoke(init_state, {"db": data["faiss_db"], "model": model})
    docs = data["faiss_db"].similarity_search(adj_q, k=6)
    if is_tbl_q:
        table_docs = [d for d in docs if "|" in d.page_content and "\n" in d.page_content]
        docs = table_docs + [d for d in docs if d not in table_docs]
    ans = load_qa_chain(model, chain_type="stuff")({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.subheader("Answer")
    st.write(ans.get("output_text", ""))
    if show_context:
        st.subheader("Retrieved Context")
        st.write(docs)

elif not pdf_file:
    st.warning("📂 Please upload a PDF.")
