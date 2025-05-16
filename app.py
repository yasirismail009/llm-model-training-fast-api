from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import PyPDF2
import pinecone
from sentence_transformers import SentenceTransformer
import shutil
import time
from typing import List
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_pinecone import PineconeVectorStore  # New import for Pinecone
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Get API tokens from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")

app = FastAPI()

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
print("====== chunking done")
print(pc.list_indexes())

# Initialize vectorstore - using PineconeVectorStore from langchain_pinecone
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

# Create custom prompt template
template = """You are a helpful AI assistant that answers questions based on the provided context from multiple documents.
Your task is to:
1. Analyze all the provided context carefully
2. Combine information from different sources if relevant
3. Provide a comprehensive and accurate answer
4. If the answer is not in the context, say "I don't have enough information to answer this question"
5. Be precise and concise in your response

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def process_pdf(file_path: str) -> List[str]:
    """Extract text from PDF and split into chunks."""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    # Split text into chunks using LangChain's text splitter
    chunks = text_splitter.split_text(text)
    return chunks

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the PDF
    chunks = process_pdf(file_path)
    
    # Add to vector store with metadata
    texts = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadatas.append({
            "source": file.filename,
            "chunk_id": i
        })
        ids.append(f"{file.filename}_{i}_{str(uuid.uuid4())}")
    
    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )
    
    return {"message": f"PDF {file.filename} processed and stored successfully"}

@app.get("/list-documents/")
async def list_documents():
    """List all documents in the database."""
    # Get all documents from Pinecone
    pinecone_index = pc.Index(INDEX_NAME)
    # Use query instead of fetch for Pinecone v2
    try:
        query_response = pinecone_index.query(
            vector=[0] * 384,  # Dummy vector for metadata-only query
            top_k=10000,
            include_metadata=True
        )
        
        # Get unique document sources
        sources = set()
        for match in query_response.matches:
            if match.metadata and "source" in match.metadata:
                sources.add(match.metadata["source"])
        return {"documents": list(sources)}
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        return {"documents": [], "error": str(e)}

@app.post("/ask/")
async def ask_question(question: str):
    """Search for the most relevant text chunk in the vector database."""
    # Get the single most relevant document from vector store
    docs = vectorstore.similarity_search(
        question,
        k=1  # Get only the best match
    )
    
    if not docs:
        return {
            "result": None,
            "source": None
        }
    
    # Get the best matching document
    best_match = docs[0]
    
    return {
        "result": best_match.page_content,
        "source": best_match.metadata["source"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)