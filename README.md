# Legal AI Assistant

An AI-powered Indian Legal Assistant built using Flask, Groq LLMs, ChromaDB, BM25 retrieval, and multilingual support.

Live Demo:  
:contentReference[oaicite:0]{index=0}

---

## Features

- Indian law focused AI assistant
- Supports BNS / BNSS / BSA 2023 legal framework
- Multilingual support
  - English
  - Hindi
  - Bengali
  - Telugu
  - Marathi
  - Tamil
  - Urdu
  - Gujarati
  - Kannada
  - Punjabi
- Hybrid Retrieval System
  - Dense vector search (ChromaDB)
  - BM25 sparse retrieval
  - Reciprocal Rank Fusion
- Legal intent classification
- Emergency legal guidance
- Legal document analysis
- OCR support for scanned PDFs/images
- Interactive MCQ clarification flow
- Landmark judgment retrieval
- Hinglish legal term normalization
- Groq + Gemini fallback support

---

## Tech Stack

### Backend
- Flask
- Python
- ChromaDB
- BM25
- LangChain

### AI Models
- Groq LLMs
- Google Gemini fallback
- HuggingFace sentence transformers

### Frontend
- HTML
- TailwindCSS
- JavaScript

---

## Project Structure

```bash
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── legal_db/
├── bm25_index.pkl
├── chroma.sqlite3
├── create_database.py
├── process_pdfs.py
└── evaluation.py
Disclaimer

This project provides general legal information for educational purposes only and should not be considered professional legal advice. Always consult a qualified advocate for legal matters.
