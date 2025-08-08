import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

import os
import tempfile

# Set your API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else st.text_input("Enter your Google API key", type="password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Define app title
st.title("📄 Smart PDF QA with LangGraph + Gemini + FAISS")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Caching extracted documents
@st.cache_data(show_spinner="Extracting and splitting PDF...")
def process_pdf(file_path):
    # Try different loaders
    for loader_cls in [PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader]:
        try:
            loader = loader_cls(file_path)
            documents = loader.load()
            if documents:
                break
        except:
            continue
    else:
        st.error("❌ Failed to load the PDF with available loaders.")
        return None, None

    # Use appropriate splitter
    if any("\n" in doc.page_content for doc in documents):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    else:
        splitter = CharacterTextSplitter(separator=". ", chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(documents)

    # Embeddings & FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    return documents, vectorstore

# Step 2: Process the PDF
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    documents, vectorstore = process_pdf(pdf_path)

    if documents and vectorstore:
        # Step 3: Accept user question
        question = st.text_input("Ask a question based on the PDF")

        if question:
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(question)

            # Step 4: LangGraph State + LLM
            class GraphState(TypedDict):
                question: str
                docs: Annotated[Sequence[Document], lambda x: x[:4]]
                messages: Annotated[Sequence[AIMessage | HumanMessage], lambda x: x[-10:]]

            def retrieve(state):
                return {"docs": retriever.get_relevant_documents(state["question"])}

            def generate(state):
                llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
                qa_chain = load_qa_chain(llm, chain_type="stuff")
                answer = qa_chain.run(input_documents=state["docs"], question=state["question"])
                if "Answer is not available in the context" in answer or not answer.strip():
                    answer = "Answer is not available in the context."
                return {"messages": [AIMessage(content=answer)]}

            builder = StateGraph(GraphState)
            builder.add_node("retrieve", retrieve)
            builder.add_node("generate", generate)
            builder.set_entry_point("retrieve")
            builder.add_edge("retrieve", "generate")
            builder.add_edge("generate", END)
            graph = builder.compile()

            # Step 5: Run graph
            inputs = {"question": question, "messages": []}
            result = graph.invoke(inputs)
            final_answer = result["messages"][-1].content
            st.success("✅ Answer:")
            st.write(final_answer)
    else:
        st.warning("Please upload a valid PDF to continue.")
