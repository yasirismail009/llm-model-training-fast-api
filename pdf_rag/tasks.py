from celery import shared_task
from django.conf import settings
import os
import PyPDF2
import pinecone
import uuid
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from .models import PDFDocument, DocumentChunk
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logger = logging.getLogger("pdf_rag")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize Pinecone
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
    text_key="text",
    namespace="myspace"
)

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# Create custom prompt template
template = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.
Your task is to:
1. Analyze all the provided context carefully
2. Combine information from different sources if relevant
3. Provide a comprehensive and accurate answer
4. If the answer is not in the context, say "I don't have enough information to answer this question based on the documents provided"
5. NEVER make up information or include any information not found in the provided context
6. NEVER search the web or use external knowledge - ONLY use the provided context
7. DO NOT mention or cite page numbers in your answer, even if page numbers are present in the context
8. The context may contain fragments of text from a PDF document, so you may need to piece together information from multiple chunks

Context:
{context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def process_pdf_page(args):
    """Process a single PDF page in a separate thread"""
    page_num, page, total_pages = args
    
    page_text = page.extract_text()
    if not page_text or not page_text.strip():
        return []  # Skip empty pages
        
    page_chunks = text_splitter.split_text(page_text)
    
    result = []
    for i, chunk in enumerate(page_chunks):
        result.append({
            "text": chunk,
            "page_num": page_num + 1,
            "total_pages": total_pages,
            "chunk_on_page": i + 1
        })
    
    return result

@shared_task
def process_pdf_task(file_path: str, filename: str, document_id: str):
    """Process a PDF file and add it to the vector database"""
    try:
        # Update document status to processing
        document = PDFDocument.objects.get(id=document_id)
        document.status = "processing"
        document.save()
        
        # Process the PDF
        chunks_with_metadata = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Process pages in parallel
            page_args = [(page_num, page, total_pages) for page_num, page in enumerate(pdf_reader.pages)]
            
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(executor.map(process_pdf_page, page_args))
            
            for page_chunks in results:
                chunks_with_metadata.extend(page_chunks)
        
        if not chunks_with_metadata:
            document.status = "failed"
            document.error_message = "No valid text chunks were extracted from the PDF"
            document.save()
            return
        
        # Process in batches
        batch_size = 100
        total_chunks = 0
        
        for i in range(0, len(chunks_with_metadata), batch_size):
            batch = chunks_with_metadata[i:i+batch_size]
            
            texts = []
            metadatas = []
            ids = []
            
            for chunk_data in batch:
                chunk_text = chunk_data["text"]
                if not chunk_text.strip():
                    continue
                    
                vector_id = f"{document_id}_p{chunk_data['page_num']}_{i}_{str(uuid.uuid4())}"
                
                texts.append(chunk_text)
                metadatas.append({
                    "source": filename,
                    "doc_id": document_id,
                    "chunk_id": i,
                    "page": chunk_data["page_num"],
                    "total_pages": chunk_data["total_pages"],
                    "chunk_on_page": chunk_data["chunk_on_page"]
                })
                ids.append(vector_id)
                
                # Create DocumentChunk
                DocumentChunk.objects.create(
                    document=document,
                    chunk_text=chunk_text,
                    page_number=chunk_data["page_num"],
                    chunk_index=chunk_data["chunk_on_page"],
                    vector_id=vector_id
                )
            
            if texts:
                vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                total_chunks += len(texts)
        
        # Update document status
        document.status = "complete"
        document.total_pages = chunks_with_metadata[-1]["total_pages"] if chunks_with_metadata else 0
        document.total_chunks = total_chunks
        document.save()
        
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        document = PDFDocument.objects.get(id=document_id)
        document.status = "failed"
        document.error_message = str(e)
        document.save()

@shared_task
def reindex_documents_task():
    """Re-process and reindex all documents"""
    try:
        # Get all documents
        documents = PDFDocument.objects.all()
        
        # Clear existing index
        pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        pinecone_index.delete(delete_all=True, namespace="myspace")
        
        # Clear document chunks
        DocumentChunk.objects.all().delete()
        
        # Process each document
        for document in documents:
            if document.file:
                process_pdf_task.delay(
                    document.file.path,
                    document.filename,
                    document.id
                )
        
        return {
            "message": f"Reindexing {documents.count()} documents",
            "files_scheduled": documents.count()
        }
    except Exception as e:
        logger.error(f"Error reindexing documents: {str(e)}")
        raise

@shared_task
def delete_document_task(document_id: str):
    """Delete a specific document"""
    try:
        document = PDFDocument.objects.get(id=document_id)
        
        # Delete from Pinecone
        pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        
        # Find all vectors with this document ID
        query_response = pinecone_index.query(
            vector=[0] * 384,
            top_k=10000,
            include_metadata=True,
            filter={"doc_id": {"$eq": document_id}},
            namespace="myspace"
        )
        
        # Delete in batches
        ids_to_delete = [match.id for match in query_response.matches]
        batch_size = 1000
        chunks_deleted = 0
        
        for i in range(0, len(ids_to_delete), batch_size):
            batch_ids = ids_to_delete[i:i+batch_size]
            if batch_ids:
                pinecone_index.delete(ids=batch_ids, namespace="myspace")
                chunks_deleted += len(batch_ids)
        
        # Delete the file if it exists
        if document.file and os.path.exists(document.file.path):
            os.remove(document.file.path)
        
        # Delete the document (this will also delete associated chunks)
        document.delete()
        
        return {
            "message": f"Document {document_id} deleted successfully",
            "chunks_deleted": chunks_deleted
        }
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise

@shared_task
def clear_all_documents_task():
    """Delete all documents"""
    try:
        # Delete all vectors from Pinecone
        pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)
        pinecone_index.delete(delete_all=True, namespace="myspace")
        
        # Delete all files
        documents = PDFDocument.objects.all()
        file_count = 0
        
        for document in documents:
            if document.file and os.path.exists(document.file.path):
                os.remove(document.file.path)
                file_count += 1
        
        # Clear database
        db_count = documents.count()
        documents.delete()
        
        return {
            "message": "All documents deleted successfully",
            "files_deleted": file_count,
            "database_records_deleted": db_count
        }
    except Exception as e:
        logger.error(f"Error clearing documents: {str(e)}")
        raise

@shared_task
def ask_question_task(question: str, doc_ids: list = None, k: int = 10, include_sources: bool = True):
    """Get an LLM-generated answer"""
    try:
        search_kwargs = {"k": k}
        if doc_ids:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}
        
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": QA_CHAIN_PROMPT,
                "verbose": True
            }
        )
        
        result = qa_chain({"query": question})
        
        response = {
            "answer": result["result"]
        }
        
        if include_sources:
            sources = []
            for doc in result["source_documents"]:
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "doc_id": doc.metadata.get("doc_id", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "total_pages": doc.metadata.get("total_pages", "Unknown"),
                    "snippet": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
                })
            response["sources"] = sources
        
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise 