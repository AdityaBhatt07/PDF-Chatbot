# 📄 Ask Your PDF 💬

A simple and powerful **Streamlit-based RAG (Retrieval-Augmented Generation) application** that allows users to upload a PDF and ask questions about its content using **Google Gemini AI**.

---

##  Features

*  Upload any PDF file
*  Extract and process text automatically
*  Smart question-answering using Gemini
*  Chat interface with memory
*  Fast retrieval using FAISS vector database

---

##  Tech Stack

* **Frontend/UI:** Streamlit
* **PDF Processing:** PyPDF2
* **LLM:** Google Gemini (via LangChain)
* **Embeddings:** Gemini Embeddings
* **Vector Database:** FAISS
* **Framework:** LangChain

---

##  Project Structure

```id="sk0o1f"
.
├── app.py
├── requirements.txt
└── README.md
```

---

##  Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ask-your-pdf.git
cd ask-your-pdf
```

---

### 2️⃣ Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Setup API Key

This project uses **Google Gemini API**.

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

##  Run the App

```bash
streamlit run app.py
```

App will open in your browser:

```id="k9u2l0"
http://localhost:8501
```

---

##  How It Works

1. Upload a PDF
2. Extract text using PyPDF2
3. Split text into chunks
4. Convert chunks into embeddings
5. Store embeddings in FAISS
6. Retrieve relevant chunks based on user query
7. Generate answer using Gemini LLM

---

##  Architecture

```id="8c3r0z"
PDF → Text → Chunks → Embeddings → FAISS
                                 ↓
User Query → Retriever → Context → LLM → Answer
```

---

##  Example Use Cases

*  Study notes Q&A
*  Research paper analysis
*  Resume understanding
*  Book summarization

---

##  Limitations

*  Does not work well with scanned PDFs (no OCR)
*  Depends on API usage limits
*  Large PDFs may take time to process

---

##  Future Improvements

* Add OCR support (for scanned PDFs)
* Show source references in answers
* Multi-PDF support
