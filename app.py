# app_online_only_gemini.py
# Fully Online RAG Chatbot: PDFs + FAISS + Gemini (1–12 students)

import os, re, requests
from datetime import datetime
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import google.generativeai as genai

# LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# -----------------------------
# PDF / text utilities
# -----------------------------
def clean_text(text: str):
    if not text:
        return ""
    text = text.lower().replace("\n", " ").replace("\r", " ").strip()
    text = " ".join(text.split())
    text = re.sub(r"\d+_prelims\.indd", "", text)
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} (am|pm)", "", text)
    text = re.sub(r"chapter \d+\.indd", "", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text

def load_pdfs_from_directory(directory_path: str):
    if not directory_path or not os.path.isdir(directory_path):
        return []
    return sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(".pdf")])

def extract_text_from_pdfs(pdf_paths):
    all_text = []
    for path in pdf_paths:
        try:
            with open(path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        cleaned = clean_text(text)
                        if cleaned:
                            all_text.append({"content": cleaned, "source": os.path.basename(path)})
        except Exception as e:
            st.warning(f"Could not read '{os.path.basename(path)}': {e}")
    return all_text

def chunk_text_with_source(text_with_source, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", ","]
    )
    chunks = []
    for item in text_with_source:
        text_chunks = splitter.split_text(item["content"])
        for c in text_chunks:
            chunks.append({"content": c, "source": item["source"]})
    return chunks

# -----------------------------
# FAISS vector store
# -----------------------------
def build_faiss_index(folder_path, index_dir):
    pdf_data = extract_text_from_pdfs(load_pdfs_from_directory(folder_path))
    if not pdf_data:
        raise ValueError("No extractable text found in PDFs.")

    chunks = chunk_text_with_source(pdf_data)
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vs = FAISS.from_texts(
        [c["content"] for c in chunks],
        embedding=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks]
    )
    vs.save_local(index_dir)
    return vs

def load_faiss_index(index_dir):
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

@st.cache_data(show_spinner=False)
def get_faiss_index(folder_path, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        return load_faiss_index(index_dir)
    return build_faiss_index(folder_path, index_dir)

# -----------------------------
# Internet check
# -----------------------------
def is_online():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False

# -----------------------------
# Online QA using Gemini
# -----------------------------
def ask_question_online(query, retriever):
    # Search PDFs first
    docs = retriever.get_relevant_documents(query)
    context = " ".join([f"{d.page_content} (from {d.metadata.get('source','unknown')})" for d in docs])

    # Prompt: Use PDF content if available, else search online for educational content only
    prompt_text = f"""
You are a kind and patient teacher for students from grade 1 to 12 (rural-friendly). 
By default answer the student's question in simple, easy-to-understand language, using short bullet points and examples wherever possible. 

Instructions:
1. Use the context provided from the PDFs to answer the question.
2. If the answer is not found in the PDF context:
   a) Clearly indicate that the content was not found in the NCERT book.
   b) Fetch the answer from educational resources online suitable for school students.
   c) Begin your answer with a note like: "Note: This answer was fetched online as it was not found in the NCERT book."
3. Only provide educational content appropriate for school students. 
4. If the question is not educational or the content cannot be found online also, respond: "Sorry, I don't know."

Context from PDFs:
{context}

Student Question:
{query}

Teacher Answer:
"""
    genai.configure(api_key="AIzaSyB223m5jS1-WlHv6zz1nT491eD4yaMl5Yg")  # <-- Replace with your key
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt_text)
    return {"answer": response.text, "model_used": "Gemini"}

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="NCERT Online Tutor", page_icon="📚")
    st.title("📚 NCERT Online Tutor: PDFs + Gemini + FAISS")

    st.session_state.setdefault("conversation_history", [])

    folder_path = r"C:\Users\PARTH\Downloads\hecu1dd"
    index_dir = r"C:\Users\PARTH\OneDrive\Desktop\sih\RAG-PDF-CHATBOT\faiss_index"

    db = get_faiss_index(folder_path, index_dir)

    st.sidebar.header("Settings")
    if st.sidebar.button("♻️ Reset History"):
        st.session_state.conversation_history.clear()
        st.success("Conversation history cleared.")

    question = st.text_input("Type your question about NCERT PDFs or school topics")
    if question:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        if is_online():
            response = ask_question_online(question, retriever)
        else:
            st.warning("No internet connection. This app requires online access.")
            return

        pdf_names = ", ".join([os.path.basename(p) for p in load_pdfs_from_directory(folder_path)])
        st.session_state.conversation_history.append(
            (question, response['answer'], response['model_used'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pdf_names)
        )

        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot ({response['model_used']}):** {response['answer']}")

    if st.session_state.conversation_history:
        st.divider()
        st.subheader("Conversation History")
        for q, a, model, ts, names in reversed(st.session_state.conversation_history):
            with st.expander(f"🕒 {ts} — {q}"):
                st.write(a)
                st.caption(f"Model: {model} | PDFs: {names}")

        df = pd.DataFrame(st.session_state.conversation_history,
                          columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        st.download_button("Download History (CSV)", data=df.to_csv(index=False), file_name="conversation_history.csv", mime="text/csv")

if __name__ == "__main__":
    main()
