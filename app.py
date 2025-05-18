from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.middleware.cors import CORSMiddleware
import os
import PyPDF2
import pinecone
from sentence_transformers import SentenceTransformer
import shutil
import time
from typing import List, Dict, Any, Optional
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Get Hugging Face API token
HUGGINGFACE_API_TOKEN =  os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")

app = FastAPI(title="PDF RAG System", 
              description="A Retrieval Augmented Generation system for querying your PDF documents with no web data")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = "pdf_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
INDEX_NAME = "methodology-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")

# Set environment variable for libraries that might use it
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and is ready
try:
    index_info = pc.describe_index(INDEX_NAME)
    if not index_info.status['ready']:
        print("Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(5)
except Exception as e:
    print(f"Index check error: {str(e)}")
    # Create index if it doesn't exist
    EMBEDDING_DIM = 384  # For all-MiniLM-L6-v2
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Created new index: {INDEX_NAME}")
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for new index to be ready...")
            time.sleep(5)

# Initialize embeddings - matches index dimension
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embeddings model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load embeddings model: {str(e)}")
    raise

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
print("✅ Text chunking configured")

# Initialize vectorstore using PineconeVectorStore
try:
    # Create the vectorstore using the new PineconeVectorStore class
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text",
        namespace="myspace"
    )
    print("✅ Vectorstore initialized successfully")
except Exception as e:
    error_msg = f"Vectorstore initialization failed: {str(e)}"
    print(f"❌ {error_msg}")
    raise RuntimeError(error_msg)

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# Create custom prompt template for the RAG chain
template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.
Your task is to:
1. Analyze all the provided context carefully
2. Combine information from different sources if relevant
3. Look for document page numbers in the context (format: [Page X of Y]) and use those in your answer
4. Provide a comprehensive and accurate answer
5. If the answer is not in the context, say "I don't have enough information to answer this question based on the documents provided" and suggest what additional information might be needed
6. NEVER make up information, hallucinate, or include any information not found in the provided context
7. NEVER search the web or use external knowledge - ONLY use the provided context
8. When providing your answer, cite specific page numbers where information was found, like this: (Page X)
9. The context may contain fragments of text from a PDF document, so you may need to piece together information from multiple chunks

Context:
{context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def process_pdf(file_path: str) -> List[dict]:
    """Extract text from PDF and split into chunks with accurate page metadata."""
    chunks_with_metadata = []
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Process each page individually to maintain page number metadata
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if not page_text or not page_text.strip():
                    continue  # Skip empty pages
                    
                # Add page number and total pages to the text for better context
                page_header = f"[Page {page_num+1} of {total_pages}]\n\n"
                page_text_with_header = page_header + page_text
                
                # Split this page into chunks
                page_chunks = text_splitter.split_text(page_text_with_header)
                
                # Add metadata for each chunk
                for i, chunk in enumerate(page_chunks):
                    chunks_with_metadata.append({
                        "text": chunk,
                        "page_num": page_num + 1,  # 1-indexed page number
                        "total_pages": total_pages,
                        "chunk_on_page": i + 1
                    })
                    
    except Exception as e:
        print(f"Error processing PDF {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    
    if not chunks_with_metadata:
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF. It may be scanned or protected.")
    
    return chunks_with_metadata

@app.post("/upload-pdf/", status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file, adding it to the vector database."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process the PDF
        chunks_with_metadata = process_pdf(file_path)
        
        # Add to vector store with metadata
        texts = []
        metadatas = []
        ids = []
        
        for i, chunk_data in enumerate(chunks_with_metadata):
            chunk_text = chunk_data["text"]
            
            # Skip empty chunks
            if not chunk_text.strip():
                continue
                
            texts.append(chunk_text)
            metadatas.append({
                "source": file.filename,
                "chunk_id": i,
                "page": chunk_data["page_num"],
                "total_pages": chunk_data["total_pages"],
                "chunk_on_page": chunk_data["chunk_on_page"]
            })
            ids.append(f"{file.filename}_p{chunk_data['page_num']}_{i}_{str(uuid.uuid4())}")
        
        if not texts:
            raise HTTPException(status_code=400, detail="No valid text chunks were extracted from the PDF")
        
        # Add chunks to vectorstore
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "message": f"PDF {file.filename} processed and stored successfully",
            "filename": file.filename,
            "chunks_extracted": len(texts),
            "pages_processed": chunks_with_metadata[-1]["total_pages"] if chunks_with_metadata else 0
        }
    except Exception as e:
        # Clean up the file if processing failed
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/list-documents/")
async def list_documents():
    """List all documents in the vector database with their metadata."""
    # Get all documents from Pinecone
    pinecone_index = pc.Index(INDEX_NAME)
    
    try:
        # Use query instead of fetch for Pinecone v2
        query_response = pinecone_index.query(
            vector=[0] * 384,  # Dummy vector for metadata-only query
            top_k=10000,
            include_metadata=True,
            namespace="myspace"
        )
        
        # Get unique document sources and count chunks per document
        sources = {}
        for match in query_response.matches:
            if match.metadata and "source" in match.metadata:
                source = match.metadata["source"]
                if source in sources:
                    sources[source] += 1
                else:
                    sources[source] = 1
        
        # Convert to list of objects with additional metadata
        documents = [
            {"filename": source, "chunks": count} 
            for source, count in sources.items()
        ]
        
        return {"documents": documents, "total_count": len(documents)}
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        return {"documents": [], "error": str(e)}

@app.post("/reindex-documents/")
async def reindex_documents():
    """Re-process and reindex all documents from the upload directory."""
    try:
        # Get list of all PDF files in the upload directory
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
        
        if not pdf_files:
            return {"message": "No PDF files found in upload directory", "count": 0}
            
        # Clear existing index
        pinecone_index = pc.Index(INDEX_NAME)
        pinecone_index.delete(delete_all=True, namespace="myspace")
        
        # Process each PDF file
        processed_files = []
        total_chunks = 0
        
        for filename in pdf_files:
            file_path = os.path.join(UPLOAD_DIR, filename)
            
            # Process the PDF
            chunks_with_metadata = process_pdf(file_path)
            
            # Add to vector store
            texts = []
            metadatas = []
            ids = []
            
            for i, chunk_data in enumerate(chunks_with_metadata):
                chunk_text = chunk_data["text"]
                
                # Skip empty chunks
                if not chunk_text.strip():
                    continue
                    
                texts.append(chunk_text)
                metadatas.append({
                    "source": filename,
                    "chunk_id": i,
                    "page": chunk_data["page_num"],
                    "total_pages": chunk_data["total_pages"],
                    "chunk_on_page": chunk_data["chunk_on_page"]
                })
                ids.append(f"{filename}_p{chunk_data['page_num']}_{i}_{str(uuid.uuid4())}")
            
            if texts:
                # Add chunks to vectorstore
                vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                processed_files.append({
                    "filename": filename,
                    "chunks": len(texts),
                    "pages": chunks_with_metadata[-1]["total_pages"] if chunks_with_metadata else 0
                })
                total_chunks += len(texts)
        
        return {
            "message": f"Successfully reindexed {len(processed_files)} documents with {total_chunks} total chunks",
            "processed_files": processed_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reindexing documents: {str(e)}")

@app.post("/retrieve-more-context/")
async def retrieve_more_context(
    question: str = Body(..., embed=True),
    document_source: Optional[str] = Body(None, embed=True),
    k: int = Body(20, embed=True)
):
    """Retrieve additional context from a specific document or all documents."""
    try:
        search_kwargs = {"k": k}
        
        # Add filter if document_source is specified
        if document_source:
            search_kwargs["filter"] = {"source": {"$eq": document_source}}
            
        # Get the relevant documents from vector store
        docs = vectorstore.similarity_search(
            question,
            **search_kwargs
        )
        
        if not docs:
            return {
                "chunks": [],
                "message": "No relevant chunks found"
            }
        
        # Format the results with more comprehensive metadata
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "total_pages": doc.metadata.get("total_pages", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "Unknown")
            })
        
        return {
            "chunks": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving additional context: {str(e)}")

@app.post("/ask-llm/")
async def ask_llm(question: str = Body(..., embed=True),
                 k: int = Body(7, embed=True),  # Increased default from 5 to 7
                 include_sources: bool = Body(True, embed=True)):
    """
    Get an LLM-generated answer based on relevant document context.
    This endpoint combines relevant chunks and uses the LLM to generate a response.
    """
    try:
        # Create retriever with specific k value - remove fetch_k which is unsupported
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create improved QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",  # Combines all documents into one context
            retriever=retriever,
            return_source_documents=True,  # Return the source documents
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "verbose": True  # For debugging
            }
        )
        
        # Get answer
        result = qa_chain({"query": question})
        
        response = {
            "answer": result["result"]
        }
        
        if include_sources:
            # Extract source information with more detailed metadata
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "total_pages": doc.metadata.get("total_pages", "Unknown"),
                    # Include a longer snippet for better context
                    "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                })
            response["sources"] = sources
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.delete("/document/{filename}")
async def delete_document(filename: str):
    """Delete a specific document from the vector database and file system."""
    try:
        # Delete from Pinecone
        pinecone_index = pc.Index(INDEX_NAME)
        
        # First, find all vectors with this source in metadata
        query_response = pinecone_index.query(
            vector=[0] * 384,  # Dummy vector for metadata-only query
            top_k=10000,
            include_metadata=True,
            filter={"source": {"$eq": filename}},
            namespace="myspace"
        )
        
        # Extract IDs to delete
        ids_to_delete = [match.id for match in query_response.matches]
        
        if ids_to_delete:
            # Delete the vectors
            pinecone_index.delete(ids=ids_to_delete, namespace="myspace")
        
        # Delete the physical file if it exists
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return {
            "message": f"Document {filename} deleted successfully",
            "chunks_deleted": len(ids_to_delete)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.delete("/clear-all-documents/")
async def clear_all_documents():
    """Delete all documents from the vector database and file system."""
    try:
        # Delete all vectors from Pinecone
        pinecone_index = pc.Index(INDEX_NAME)
        pinecone_index.delete(delete_all=True, namespace="myspace")
        
        # Delete all files in upload directory
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return {"message": "All documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)