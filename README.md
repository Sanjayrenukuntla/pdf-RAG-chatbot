# PDF RAG Chatbot

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. Upload a PDF, ask questions about its contents, and receive context-aware answers powered by Google's Gemini.

## Overview

This project allows users to interact with PDF documents through a conversational interface.

Instead of sending an entire document to an AI model, the application:
1. Extracts text from the uploaded PDF.
2. Splits the text into smaller, overlapping chunks.
3. Generates vector embeddings using Sentence Transformers.
4. Stores embeddings in ChromaDB.
5. Retrieves relevant document sections based on the user's question.
6. Sends the retrieved context to Gemini to generate an answer.

The chatbot is designed to answer using the uploaded document rather than inventing information when relevant context is unavailable.

## Features

- PDF document uploading and processing.
- Automatic text extraction and chunking.
- Semantic search using vector embeddings.
- Persistent vector storage with ChromaDB.
- Context-aware question answering using Google Gemini.
- Conversational interface built with Streamlit.
- Chat history maintained during the session.
- Responses grounded in retrieved PDF content.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Streamlit | User interface |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| Google Gemini | Answer generation |
| PyPDF | PDF text extraction |
| OpenAI Python SDK | Gemini-compatible API client |

## Architecture

```text
             User uploads PDF
                    |
                    v
              PDF Extraction
                    |
                    v
                Text Chunking
                    |
                    v
          Sentence Transformer
             Embeddings
                    |
                    v
               ChromaDB
             Vector Storage
                    |
                    |
User asks a question
          |
          v
   Question Embedding
          |
          v
 Semantic Retrieval
          |
          v
 Relevant PDF Chunks
          |
          v
     Gemini API
          |
          v
   Generated Answer
          |
          v
      Streamlit UI
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sanjayrenukuntla/pdf-RAG-chatbot.git
cd pdf-RAG-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root.

Add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key
```

Get an API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Never upload your `.env` file or expose your API key publicly.**

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Project Structure

```text
pdf-RAG-chatbot/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── screenshots/
```

## Screenshots

Screenshots demonstrating the application's interface and outputs will be added here.

| Application | Description |
|---|---|
| Homepage | PDF upload interface |
| PDF Processing | Document processing status |
| Question Answering | AI-generated answers |
| Context Retrieval | Answers grounded in PDF content |
| Missing Information | Handling questions outside the document |

## Current Limitations

- Supports PDF documents with extractable text; scanned PDFs requiring OCR are not currently supported.
- Processes one active PDF at a time.
- Answer quality depends on document quality and retrieval relevance.
- Requires an internet connection and a valid Gemini API key for answer generation.

## Future Improvements

- Support multiple PDF documents.
- Add OCR for scanned documents.
- Display retrieved source passages and page numbers.
- Add document-level citations.
- Introduce hybrid search and improved retrieval strategies.
- Explore RAG evaluation and response-quality metrics.
- Deploy the application online.

## Author

**Sanjay Renukuntla**

[GitHub](https://github.com/Sanjayrenukuntla)

---

*Built as a hands-on project to explore Retrieval-Augmented Generation, semantic search, vector databases, and LLM-powered applications.*
