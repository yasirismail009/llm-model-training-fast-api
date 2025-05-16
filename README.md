# PDF Question Answering System

This project allows you to upload PDF documents, process them, and ask questions about their content using AI-powered semantic search.

## Features

- PDF document upload and processing
- Vector database storage using ChromaDB
- Semantic search for question answering
- FastAPI backend with CORS support

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### Upload PDF
- **POST** `/upload-pdf/`
- Upload a PDF file to be processed and stored in the vector database
- Accepts multipart/form-data with a PDF file

### Ask Questions
- **POST** `/ask/`
- Ask questions about the uploaded PDFs
- Returns relevant answers with source PDF information

## Example Usage

1. Upload a PDF:
```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:8000/upload-pdf/
```

2. Ask a question:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"question":"What is the main topic of the document?"}' http://localhost:8000/ask/
```

## Project Structure

- `pdf_uploads/`: Directory for storing uploaded PDF files
- `vector_db/`: Directory for ChromaDB vector database
- `app.py`: Main application file
- `requirements.txt`: Project dependencies 