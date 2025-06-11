from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
import json
from rest_framework.permissions import AllowAny
from django.conf import settings
import os
import PyPDF2
import pinecone
import uuid
import shutil
from .models import PDFDocument, DocumentChunk, Tag
from .serializers import (
    PDFDocumentSerializer, DocumentUploadSerializer,
    QuerySerializer, ContextRetrievalSerializer, TagSerializer
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from .utils import process_pdf, text_splitter

load_dotenv()

# Initialize Pinecone
INDEX_NAME = "methodology-index"
NAMESPACE = "myspace"

# Initialize Pinecone client
pc = pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Get or create index
try:
    # Check if index exists
    if INDEX_NAME not in pc.list_indexes():
        # Create index if it doesn't exist
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Dimension for all-MiniLM-L6-v2
            metric="cosine"
        )
except Exception as e:
    print(f"Error initializing Pinecone index: {str(e)}")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Initialize vectorstore
vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    text_key="text",
    namespace=NAMESPACE
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

# Create custom prompt template with stronger directives
template =  """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.
Your task is to:
1. Analyze all the provided context carefully
2. Combine information from different sources if relevant
3. Look for document page numbers in the context (format: [Page X of Y]) and use those in your answer
4. Provide a comprehensive and accurate answer
5. Do compilation if there is comparison or compilation in the question
6. Do comparison if there is comparison in the question
7. Do highlights if there is highlights in the question

Context:
{context}

Question: {question}

Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

def process_pdf_chunk(args):
    """Process a chunk of PDF pages."""
    file_path, start_page, end_page, total_pages, text_splitter = args
    chunks_with_metadata = []
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Process assigned range of pages
            for page_num in range(start_page, min(end_page, total_pages)):
                page = pdf_reader.pages[page_num]
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
        # Log error but continue processing other chunks
        print(f"Error processing pages {start_page}-{end_page}: {str(e)}")
        
    return chunks_with_metadata


def process_pdf_parallel(file_path: str, text_splitter, max_workers=4, chunk_size=100):
    """Extract text from PDF and split into chunks with parallel processing."""
    all_chunks = []
    
    try:
        # First pass to get total pages
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
        
        # Skip processing if PDF is empty
        if total_pages == 0:
            return []
            
        # Create chunks of pages to process in parallel
        page_chunks = []
        for i in range(0, total_pages, chunk_size):
            page_chunks.append((file_path, i, i + chunk_size, total_pages, text_splitter))
        
        # Process chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_pdf_chunk, page_chunks))
            
        # Combine results
        for result in results:
            all_chunks.extend(result)
            
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")
    
    if not all_chunks:
        raise Exception("No text could be extracted from the PDF. It may be scanned or protected.")
    
    return all_chunks


def batch_vectorstore_upload(chunks_with_metadata, document, vectorstore, batch_size=100):
    """Upload chunks to vectorstore in batches."""
    total_processed = 0
    
    for i in range(0, len(chunks_with_metadata), batch_size):
        batch = chunks_with_metadata[i:i+batch_size]
        
        texts = []
        metadatas = []
        ids = []
        
        for j, chunk_data in enumerate(batch):
            chunk_index = i + j
            vector_id = f"{document.filename}_p{chunk_data['page_num']}_{chunk_index}_{str(uuid.uuid4())}"
            
            # Create DocumentChunk
            DocumentChunk.objects.create(
                document=document,
                chunk_text=chunk_data["text"],
                page_number=chunk_data["page_num"],
                chunk_index=chunk_data["chunk_on_page"],
                vector_id=vector_id
            )
            
            # Prepare for vectorstore batch upload
            texts.append(chunk_data["text"])
            metadatas.append({
                "source": document.filename,
                "chunk_id": chunk_index,
                "page": chunk_data["page_num"],
                "total_pages": chunk_data["total_pages"],
                "chunk_on_page": chunk_data["chunk_on_page"]
            })
            ids.append(vector_id)
        
        # Add batch to vectorstore
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        total_processed += len(batch)
        # Update progress periodically
        if total_processed % 500 == 0:
            print(f"Processed {total_processed} chunks out of {len(chunks_with_metadata)}")
            
    return total_processed


class PDFUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [AllowAny]

    def post(self, request):
        """Handle PDF upload with multipart/form-data"""
        serializer = DocumentUploadSerializer(data=request.data)
        if serializer.is_valid():
            pdf_file = serializer.validated_data['file']
            tags = serializer.validated_data.get('tags', [])
            
            # Create PDFDocument instance without tags
            document = PDFDocument.objects.create(
                filename=pdf_file.name,
                file=pdf_file
            )
            
            # Add tags after creation
            if tags:
                # Handle the case where tags is a list containing a JSON string
                if isinstance(tags, list) and len(tags) > 0 and isinstance(tags[0], str):
                    try:
                        # Parse the JSON string to get the actual tag names
                        tag_names = json.loads(tags[0])
                        for tag_name in tag_names:
                            tag, _ = Tag.objects.get_or_create(name=tag_name)
                            document.tags.add(tag)
                    except json.JSONDecodeError:
                        # If not JSON, treat each item as a tag name
                        for tag_name in tags:
                            tag, _ = Tag.objects.get_or_create(name=tag_name)
                            document.tags.add(tag)
            
            try:
                # Process PDF
                chunks_with_metadata = process_pdf(document.file.path)
                document.total_pages = chunks_with_metadata[-1]["total_pages"] if chunks_with_metadata else 0
                
                # Create chunks and add to vectorstore
                for i, chunk_data in enumerate(chunks_with_metadata):
                    vector_id = f"{document.filename}_p{chunk_data['page_num']}_{i}_{str(uuid.uuid4())}"
                    
                    # Create DocumentChunk
                    DocumentChunk.objects.create(
                        document=document,
                        chunk_text=chunk_data["text"],
                        page_number=chunk_data["page_num"],
                        chunk_index=chunk_data["chunk_on_page"],
                        vector_id=vector_id
                    )
                    
                    # Add to vectorstore with tags
                    vectorstore.add_texts(
                        texts=[chunk_data["text"]],
                        metadatas=[{
                            "source": document.filename,
                            "chunk_id": i,
                            "page": chunk_data["page_num"],
                            "total_pages": chunk_data["total_pages"],
                            "chunk_on_page": chunk_data["chunk_on_page"],
                            "tags": tags
                        }],
                        ids=[vector_id]
                    )
                    
                    document.total_chunks += 1
                
                document.save()
                
                return Response(
                    PDFDocumentSerializer(document).data,
                    status=status.HTTP_201_CREATED
                )
            except Exception as e:
                document.delete()
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_400_BAD_REQUEST
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Add these settings to your settings.py
"""
# PDF Processing Settings
LARGE_PDF_MAX_WORKERS = 8  # Number of parallel processes for large PDFs
LARGE_PDF_CHUNK_SIZE = 200  # Number of pages per worker for large PDFs
DEFAULT_PDF_MAX_WORKERS = 4  # Number of parallel processes for normal PDFs
DEFAULT_PDF_CHUNK_SIZE = 100  # Number of pages per worker for normal PDFs
VECTORSTORE_BATCH_SIZE = 100  # Number of chunks per vectorstore upload batch
"""
class DocumentListView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        """List all documents in the vector database with their metadata."""
        try:
            # Get all documents from Pinecone
            pinecone_index = pc.Index(INDEX_NAME)
            
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
            
            return Response({
                "documents": documents,
                "total_count": len(documents)
            })
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ReindexDocumentsView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        """Re-process and reindex all documents from the upload directory."""
        try:
            # Get list of all PDF files in the upload directory
            pdf_files = [f for f in os.listdir(settings.PDF_UPLOAD_DIR) if f.endswith('.pdf')]
            
            if not pdf_files:
                return Response({
                    "message": "No PDF files found in upload directory",
                    "count": 0
                })
                
            # Clear existing index
            pinecone_index = pc.Index(INDEX_NAME)
            try:
                # Try to delete the namespace
                pinecone_index.delete(delete_all=True, namespace=NAMESPACE)
            except Exception as e:
                # If namespace doesn't exist, that's fine - we'll create it when adding vectors
                print(f"Namespace deletion error (can be ignored): {str(e)}")
            
            # Process each PDF file
            processed_files = []
            total_chunks = 0
            
            for filename in pdf_files:
                file_path = os.path.join(settings.PDF_UPLOAD_DIR, filename)
                
                # Get document from database to preserve tags
                try:
                    document = PDFDocument.objects.get(filename=filename)
                    tags = document.tags
                except PDFDocument.DoesNotExist:
                    tags = []
                
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
                        "chunk_on_page": chunk_data["chunk_on_page"],
                        "tags": tags
                    })
                    ids.append(f"{filename}_p{chunk_data['page_num']}_{i}_{str(uuid.uuid4())}")
                
                if texts:
                    try:
                        # Add chunks to vectorstore
                        vectorstore.add_texts(
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )
                        
                        processed_files.append({
                            "filename": filename,
                            "chunks": len(texts),
                            "pages": chunks_with_metadata[-1]["total_pages"] if chunks_with_metadata else 0,
                            "tags": tags
                        })
                        total_chunks += len(texts)
                    except Exception as e:
                        print(f"Error processing file {filename}: {str(e)}")
                        continue
            
            return Response({
                "message": f"Successfully reindexed {len(processed_files)} documents with {total_chunks} total chunks",
                "processed_files": processed_files
            })
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RetrieveContextView(APIView):
    parser_classes = [JSONParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Retrieve additional context from a specific document or all documents."""
        serializer = ContextRetrievalSerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            document_source = serializer.validated_data.get('document_source')
            tags = serializer.validated_data.get('tags', [])
            k = serializer.validated_data['k']
            
            try:
                search_kwargs = {"k": k}
                filter_conditions = {}
                
                if document_source:
                    filter_conditions["source"] = {"$eq": document_source}
                
                if tags:
                    filter_conditions["tags"] = {"$in": tags}
                
                if filter_conditions:
                    search_kwargs["filter"] = filter_conditions
                
                docs = vectorstore.similarity_search(
                    question,
                    **search_kwargs
                )
                
                if not docs:
                    return Response({
                        "chunks": [],
                        "message": "No relevant chunks found"
                    })
                
                results = []
                for doc in docs:
                    results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "Unknown"),
                        "total_pages": doc.metadata.get("total_pages", "Unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                        "tags": doc.metadata.get("tags", [])
                    })
                
                return Response({
                    "chunks": results,
                    "count": len(results)
                })
            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AskQuestionView(APIView):
    parser_classes = [JSONParser]
    permission_classes = [AllowAny]

    def post(self, request):
        """Get an LLM-generated answer based on relevant document context."""
        serializer = QuerySerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            tags = serializer.validated_data.get('tags', [])
            
            try:
                # Create retriever with tag filtering
                search_kwargs = {"k": 10}  # Increased for better context
                if tags:
                    # Handle the case where tags is a list containing a JSON string
                    if isinstance(tags, list) and len(tags) > 0 and isinstance(tags[0], str):
                        try:
                            # Parse the JSON string to get the actual tag names
                            tag_names = json.loads(tags[0])
                            print(f"Parsed tag names: {tag_names}")
                            search_kwargs["filter"] = {"tags": {"$in": tag_names}}
                        except json.JSONDecodeError:
                            # If not JSON, use the tags as they are
                            print(f"Using raw tags: {tags}")
                            search_kwargs["filter"] = {"tags": {"$in": tags}}
                
                print(f"Search kwargs: {search_kwargs}")
                
                # Try similarity search with better error handling
                similar_docs = vectorstore.similarity_search(
                    question,
                    **search_kwargs
                )
                print(f"Found {len(similar_docs)} similar documents")
                
                if not similar_docs:
                    # If no documents found, try without tag filtering
                    print("No documents found with tags, trying without filters")
                    similar_docs = vectorstore.similarity_search(
                        question,
                        k=10  # Use same increased value
                    )
                    print(f"Found {len(similar_docs)} documents without filters")
                    
                    if similar_docs:
                        # Use these documents without filtering
                        search_kwargs = {"k": 10}
                
                # Return error if still no documents found
                if not similar_docs:
                    return Response(
                        {"error": "No relevant documents found for your query."},
                        status=status.HTTP_404_NOT_FOUND
                    )
                
                # Create retriever
                retriever = vectorstore.as_retriever(
                    search_kwargs=search_kwargs
                )
                
                # Create QA chain
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
                
                # Get answer
                result = qa_chain({"query": question})
                
                # Clean up the response better
                answer = result["result"]
                if isinstance(answer, str):
                    answer = answer.strip()
                    # Remove quotes and clean formatting
                    if answer.startswith("'") and answer.endswith("'"):
                        answer = answer[1:-1]
                    if answer.startswith('"') and answer.endswith('"'):
                        answer = answer[1:-1]
                
                response = {
                    "answer": answer
                }
                
                # Process sources with better deduplication
                sources = []
                seen_sources = set()  # To track unique source/page combinations
                
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "Unknown")
                    total_pages = doc.metadata.get("total_pages", "Unknown")
                    
                    # Create unique key for deduplication
                    source_key = f"{source}_{page}"
                    
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        
                        # Better snippet handling
                        content = doc.page_content.strip()
                        snippet = content[:300] + "..." if len(content) > 300 else content
                        
                        sources.append({
                            "source": source,
                            "page": float(page) if isinstance(page, (int, float, str)) and str(page).replace('.', '').isdigit() else page,
                            "total_pages": float(total_pages) if isinstance(total_pages, (int, float, str)) and str(total_pages).replace('.', '').isdigit() else total_pages,
                            "snippet": snippet
                        })
                
                response["sources"] = sources
                
                return Response(response)
                
            except Exception as e:
                print(f"Error in AskQuestionView: {str(e)}")
                return Response(
                    {"error": f"Processing failed: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class DeleteDocumentView(APIView):
    permission_classes = [AllowAny]

    def delete(self, request, document_id):
        """Delete a specific document from the vector database and file system."""
        try:
            document = PDFDocument.objects.get(id=document_id)
            
            # Delete from Pinecone
            pinecone_index = pc.Index(INDEX_NAME)
            
            # First, find all vectors with this source in metadata
            query_response = pinecone_index.query(
                vector=[0] * 384,  # Dummy vector for metadata-only query
                top_k=10000,
                include_metadata=True,
                filter={"source": {"$eq": document.filename}},
                namespace="myspace"
            )
            
            # Extract IDs to delete
            ids_to_delete = [match.id for match in query_response.matches]
            
            if ids_to_delete:
                # Delete the vectors
                pinecone_index.delete(ids=ids_to_delete, namespace="myspace")
            
            # Delete the document (this will also delete associated chunks)
            document.delete()
            
            return Response({
                "message": f"Document {document.filename} deleted successfully",
                "chunks_deleted": len(ids_to_delete)
            })
        except PDFDocument.DoesNotExist:
            return Response(
                {"error": "Document not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class ClearAllDocumentsView(APIView):
    permission_classes = [AllowAny]

    def delete(self, request):
        """Delete all documents from the vector database and file system."""
        try:
            # Delete all vectors from Pinecone
            pinecone_index = pc.Index(INDEX_NAME)
            pinecone_index.delete(delete_all=True, namespace="myspace")
            
            # Delete all documents and their chunks
            PDFDocument.objects.all().delete()
            
            return Response({"message": "All documents deleted successfully"})
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DocumentTagsView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def patch(self, request, document_id):
        """Update tags for a specific document."""
        try:
            document = PDFDocument.objects.get(id=document_id)
            
            # Validate tags
            if 'tags' not in request.data:
                return Response(
                    {"error": "Tags field is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            tags = request.data['tags']
            if not isinstance(tags, list):
                return Response(
                    {"error": "Tags must be a list"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Update document tags
            document.tags = tags
            document.save()
            
            # Update tags in vectorstore
            pinecone_index = pc.Index(INDEX_NAME)
            
            # Find all vectors for this document
            query_response = pinecone_index.query(
                vector=[0] * 384,  # Dummy vector for metadata-only query
                top_k=10000,
                include_metadata=True,
                filter={"source": {"$eq": document.filename}},
                namespace=NAMESPACE
            )
            
            # Update metadata for each vector
            for match in query_response.matches:
                metadata = match.metadata
                metadata['tags'] = tags
                
                # Update vector metadata
                pinecone_index.update(
                    id=match.id,
                    metadata=metadata,
                    namespace=NAMESPACE
                )
            
            return Response({
                "message": f"Successfully updated tags for document {document.filename}",
                "document_id": str(document.id),
                "tags": tags
            })
            
        except PDFDocument.DoesNotExist:
            return Response(
                {"error": "Document not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class TagListView(APIView):
    """View to list all tags and create new ones."""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]
    
    def get(self, request):
        """List all tags."""
        tags = Tag.objects.all()
        serializer = TagSerializer(tags, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        """Create a new tag."""
        serializer = TagSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TagDetailView(APIView):
    """View to retrieve, update or delete a tag."""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]
    
    def get_object(self, pk):
        try:
            return Tag.objects.get(pk=pk)
        except Tag.DoesNotExist:
            return None
    
    def get(self, request, pk):
        """Retrieve a tag."""
        tag = self.get_object(pk)
        if not tag:
            return Response(status=status.HTTP_404_NOT_FOUND)
        serializer = TagSerializer(tag)
        return Response(serializer.data)
    
    def put(self, request, pk):
        """Update a tag."""
        tag = self.get_object(pk)
        if not tag:
            return Response(status=status.HTTP_404_NOT_FOUND)
        serializer = TagSerializer(tag, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        """Delete a tag."""
        tag = self.get_object(pk)
        if not tag:
            return Response(status=status.HTTP_404_NOT_FOUND)
        tag.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
