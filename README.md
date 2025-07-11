# 📘 Math Mentor — Local RAG-Powered Math Assistant using Gemma Via Ollama

Math Mentor is a **local Retrieval-Augmented Generation (RAG)** app built with **Streamlit**, **LangChain**, and **Gemma via Ollama**. It helps you ask questions based on your own math syllabus PDFs using vector search and a local LLM.

---

## 🚀 Features

- 🔍 Ask math questions from your uploaded syllabus  
- 🧠 Uses FAISS for efficient vector retrieval  
- 🤖 Powered by HuggingFace embeddings & Gemma (via Ollama)  
- ⚡ Works locally without external APIs  
- 🧾 Easy UI with Streamlit  

---

## 🖼️ Working Screenshot

<p align="center">
  <img src="Screenshots/9e42c84d-1f28-4280-8a65-183b1d246e69.jpg" alt="Working Screenshot" width="600"/>
</p>

---

## 💻 Tech Stack
- Python: Core programming language for app logic  
- Streamlit: Web app framework for creating the UI  
- LangChain: Framework for building LLM-powered applications  
- HuggingFace Embeddings: To generate embeddings from syllabus text  
- FAISS: Facebook AI Similarity Search for vector database and retrieval  
- Ollama + Gemma: Local LLM backend for answering questions  

---

## 🛠️ Project Structure

```bash
mathmentor-main/
│
├── syllabus/
│   ├── Linear Algebra and Ordinary Differential Equations.pdf
│   └── Numerical Methods and Vector Calculus.pdf
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── Screenshots/
│   └── 9e42c84d-1f28-4280-8a65-183b1d246e69.jpg
│
├── .venv/
├── app.py
├── embed_pdfs.py
├── requirements.txt
└── README.md
