
# app.py
# Chat with Multiple PDFs from a Directory (FAISS + Gemini via LangChain)

import os
import base64
from datetime import datetime
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
import json

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader

# LangChain / Google GenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Enable persistent caching
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

CACHE_FILE = "qa_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        qa_cache = json.load(f)
else:
    qa_cache = {}

# -----------------------------
# Helpers: PDF loading & text
# -----------------------------
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(qa_cache, f)

def cached_query(query: str, chain, retriever):
    """Return cached answer if available, else run chain and save to cache."""
    if query in qa_cache:
        return qa_cache[query]

    docs = retriever.get_relevant_documents(query)
    response = chain(
        {"input_documents": docs, "question": query},
        return_only_outputs=True
    )
    answer = response["output_text"]

    qa_cache[query] = answer
    save_cache()
    return answer

def load_pdfs_from_directory(directory_path: str):
    """Return absolute paths of all .pdf files inside the directory (non-recursive)."""
    if not directory_path or not os.path.isdir(directory_path):
        return []
    pdf_paths = []
    for fname in os.listdir(directory_path):
        if fname.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(directory_path, fname))
    return sorted(pdf_paths)

def extract_text_from_pdfs(pdf_paths):
    """Read all PDFs and return a single concatenated string of text."""
    all_text = []
    for path in pdf_paths:
        try:
            with open(path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    txt = page.extract_text() or ""
                    all_text.append(txt)
        except Exception as e:
            st.warning(f"Could not read '{os.path.basename(path)}': {e}")
    return "\n".join(all_text)

def chunk_text(text: str):
    """Split text into overlapping chunks for RAG."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ".", "?", "!", ","],
    )
    return splitter.split_text(text)

# -----------------------------
# Vector store (FAISS)
# -----------------------------
def build_faiss_index_from_folder(folder_path: str, api_key: str, index_dir: str = "faiss_index"):
    """Create embeddings for all text chunks from PDFs in folder and save FAISS locally."""
    pdf_paths = load_pdfs_from_directory(folder_path)
    if not pdf_paths:
        raise ValueError("No PDFs found in the given folder.")

    raw_text = extract_text_from_pdfs(pdf_paths)
    if not raw_text.strip():
        raise ValueError("No extractable text found in PDFs.")

    chunks = chunk_text(raw_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local(index_dir)
    return vs

def load_faiss_index(api_key: str, index_dir: str = "faiss_index"):
    """Load existing FAISS index if available."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

# -----------------------------
# QA Chain
# -----------------------------
def make_qa_chain(api_key: str):
    prompt_template = """
You are a kind and patient teacher for students in classes 1–12, especially from rural areas.  
Always use **simple words** and **easy examples**.  

Rules for answering:
1. Use ONLY the information from the given context.  
2. If the answer is not in the context, say:  
   "answer is not available in the context".  
3. Always explain in **short steps or bullet points**.  
4. Give a **small example** if it helps.  
5. At the end, ask the student a follow-up question to keep them curious.  

Context:
{context}

Student’s Question:
{question}

Teacher’s Answer:
"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# -----------------------------
# UI: Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs (Folder)", page_icon="📚")
    st.title("📚 RAG + Cached AI Bot for NCERT Books")

    # session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False

    st.sidebar.header("Settings")

    api_key = os.getenv("GOOGLE_API_KEY")

    folder_path = r"C:\Users\PARTH\Downloads\hecu1dd"
    index_dir = st.sidebar.text_input("FAISS Index Directory", value="faiss_index")

    colA, colB = st.sidebar.columns(2)
    build_btn = colA.button("🔨 Build / Rebuild Index")
    reset_btn = colB.button("♻️ Reset History")

    if reset_btn:
        st.session_state.conversation_history = []
        st.success("Conversation history cleared.")

    # Build or rebuild the index
    if build_btn:
        if not api_key:
            st.warning("Please enter your Google API Key.")
        elif not folder_path.strip():
            st.warning("Please enter a valid folder path.")
        else:
            with st.spinner("Building FAISS index from your folder (this may take a moment)..."):
                try:
                    build_faiss_index_from_folder(folder_path, api_key, index_dir=index_dir)
                    st.session_state.index_ready = True
                    st.success("FAISS index built and saved successfully!")
                except Exception as e:
                    st.session_state.index_ready = False
                    st.error(f"Failed to build index: {e}")

    # Try to load an existing index if not flagged ready yet
    if not st.session_state.index_ready and api_key:
        try:
            _ = load_faiss_index(api_key, index_dir=index_dir)
            st.session_state.index_ready = True
        except Exception:
            pass  # no existing index yet

    # Ask a question
    st.subheader("Ask a question")
    question = st.text_input("Type your question about the folder PDFs")

    if question:
        if not api_key:
            st.warning("Please enter your Google API Key in the sidebar.")
        elif not st.session_state.index_ready:
            st.warning("Please build the index first (sidebar → Build / Rebuild Index).")
        else:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
                retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                chain = make_qa_chain(api_key)

                # now we directly get final answer string
                answer = cached_query(question, chain, retriever)

                # record history
                pdf_names = ", ".join([os.path.basename(p) for p in load_pdfs_from_directory(folder_path)])
                st.session_state.conversation_history.append(
                    (question, answer, "Google AI", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pdf_names)
                )

                # show chat turn
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Bot:** {answer}")

            except Exception as e:
                st.error(f"Query failed: {e}")

    # history + download
    if len(st.session_state.conversation_history) > 0:
        st.divider()
        st.subheader("Conversation History")
        for q, a, model, ts, names in reversed(st.session_state.conversation_history):
            with st.expander(f"🕒 {ts} — {q}"):
                st.write(a)
                st.caption(f"Model: {model} | PDFs: {names}")

        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"]
        )
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.download_button(
            "Download History (CSV)",
            data=csv,
            file_name="conversation_history.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()