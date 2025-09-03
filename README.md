Here’s a professional, clear, and “human-friendly” README for your **RAG + Gemini NCERT Tutor project**:

---

# 📚 NCERT Hybrid Tutor (RAG + Gemini)

A fully online **RAG-based chatbot** for students (grades 1–12) that answers questions using **NCERT PDFs** and online resources via **Google Gemini API**. Designed to provide **educational content only**, with clear, student-friendly explanations.

---

## 🔹 Features

* **RAG (Retrieval-Augmented Generation)** using PDFs + FAISS
* Answers questions using **NCERT textbooks** first
* **Fallback online search** via Gemini if content not in PDFs
* Ensures answers are **suitable for school students (grades 1–12)**
* Clearly indicates when content is fetched **from online resources**
* Conversation history with **timestamps, PDF sources, and model info**
* Downloadable conversation history as CSV
* Lightweight, fast, and simple web interface

---

## 🔹 Technologies Used

* **Backend:** Python, FastAPI
* **PDF Parsing:** PyPDF2
* **Vector Store:** FAISS via LangChain
* **LLM:** Google Gemini API (gemini-2.5-flash)
* **Frontend (Optional):** Next.js + Tailwind CSS
* **Utilities:** pandas, requests, regex for PDF cleaning

---

## 🔹 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-gemini-ncert-tutor.git
cd rag-gemini-ncert-tutor
```

### 2. Create virtual environment

```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API key

Set your **Google Gemini API key** in `app_online_only_gemini.py`:

```python
genai.configure(api_key="YOUR_API_KEY")
```

---

## 🔹 Running the App

### **FastAPI Backend**

```bash
uvicorn app_online_only_gemini:app --reload
```

API endpoint:

* `POST /ask` → Accepts JSON `{ "question": "<your question>" }`
* Returns: answer, model used, PDFs used, timestamp

### **Next.js Frontend (Optional)**

```bash
cd ncert-tutor-frontend
npm install
npm run dev
```

* Open [http://localhost:3000](http://localhost:3000)
* Ask questions and see answers with source PDFs and timestamp

---

## 🔹 How It Works

1. **PDF Extraction:** Reads all NCERT PDFs from a folder
2. **Text Cleaning & Chunking:** Converts content into small chunks with overlap
3. **FAISS Index:** Builds vector store for similarity search
4. **RAG Retrieval:** Queries top chunks from PDFs
5. **Gemini LLM:**

   * Uses PDF content first
   * If answer not found → searches online educational content only
   * Clearly marks online-fetched answers
6. **Response:** Returns structured answer suitable for grades 1–12

---

## 🔹 Usage

* Type a question related to **NCERT syllabus** or **school subjects**

* Example:

  ```
  Question: What is photosynthesis?
  ```

* Output includes:

  * Student-friendly explanation
  * Source PDFs (if found)
  * Timestamp and model used

* Non-educational or unrelated questions → responds:

  ```
  Sorry, I don't know.
  ```

---

## 🔹 Folder Structure

```
rag-gemini-ncert-tutor/
├─ app_online_only_gemini.py      # FastAPI + RAG backend
├─ requirements.txt               # Python dependencies
├─ faiss_index/                   # FAISS index for PDFs
├─ PDFs/                          # NCERT PDFs
└─ ncert-tutor-frontend/          # Optional Next.js frontend
```

---

## 🔹 Notes

* Free Gemini API may have **rate limits** → reduce chunk size to 300–500 tokens for faster queries
* Internet is **required** for online fallback answers
* Designed for **educational purposes only**

---

## 🔹 License

MIT License

---

I can also make a **shorter, GitHub-ready version** with badges, screenshots, and links if you want it to look very professional for your repo.

Do you want me to do that?