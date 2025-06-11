"""
Advanced Vector Storage Service with Organization Isolation and Multi-LLM Support

This module provides comprehensive vector storage capabilities with:
- Organization-level isolation
- Multiple LLM backend support (AWS Bedrock, Databricks, HuggingFace, etc.)
- Advanced retrieval with reranking
- Multi-query generation
- Confidence scoring
- Obligation detection
- Knowledge map creation
"""

import os
import hashlib
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
import logging

import pinecone
import numpy as np
from langchain.schema import Document as LangChainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough

# LLM Imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import DatabricksEmbeddings
from langchain_community.chat_models import ChatDatabricks

# Reranking and Advanced Retrieval
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import mlflow
from langsmith import Client as LangSmithClient

# Django imports
from django.conf import settings
from django.utils import timezone as django_timezone
from django.db import models
from .models import (
    Organization, VectorIndex, VectorChunk, QuerySession, 
    QueryLog, LLMConfiguration, KnowledgeMap, Document
)

logger = logging.getLogger(__name__)

class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for multi-query retrieval."""
    
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

class AdvancedVectorService:
    """Advanced vector storage service with organization isolation and multi-LLM support."""
    
    def __init__(self, organization: Organization):
        self.organization = organization
        self.pinecone_client = None
        self.vector_index = None
        self.embeddings = None
        self.llm = None
        self.reranker = None
        self.langsmith_client = None
        
        # Initialize services
        self._initialize_pinecone()
        self._initialize_embeddings()
        self._initialize_llm()
        self._initialize_reranker()
        self._initialize_monitoring()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client with organization-specific configuration."""
        try:
            if self.organization.pinecone_api_key:
                self.pinecone_client = pinecone.Pinecone(
                    api_key=self.organization.pinecone_api_key,
                    environment=self.organization.pinecone_environment or "us-east-1-aws"
                )
                
                # Get or create index
                index_name = self.organization.pinecone_index_name or f"org-{self.organization.slug}"
                
                if index_name not in self.pinecone_client.list_indexes():
                    # Create index if it doesn't exist
                    self.pinecone_client.create_index(
                        name=index_name,
                        dimension=384,  # Default for all-MiniLM-L6-v2
                        metric="cosine",
                        spec=pinecone.ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                
                self.vector_index = self.pinecone_client.Index(index_name)
                
                # Update or create VectorIndex record
                vector_index_obj, created = VectorIndex.objects.get_or_create(
                    organization=self.organization,
                    name=index_name,
                    defaults={
                        'index_type': 'pinecone',
                        'dimension': 384,
                        'metric': 'cosine'
                    }
                )
                
                logger.info(f"Pinecone initialized for organization {self.organization.name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model based on organization configuration."""
        try:
            model_name = self.organization.embedding_model_name
            
            if self.organization.llm_provider == 'aws_bedrock':
                self.embeddings = BedrockEmbeddings(
                    model_id="amazon.titan-embed-text-v1",
                    region_name=self.organization.aws_region,
                    credentials_profile_name=None
                )
            elif self.organization.llm_provider == 'databricks':
                self.embeddings = DatabricksEmbeddings(
                    endpoint=f"{self.organization.databricks_host}/serving-endpoints/databricks-bge-large-en",
                    databricks_token=self.organization.databricks_token
                )
            elif self.organization.llm_provider == 'openai':
                self.embeddings = OpenAIEmbeddings(
                    model=model_name or "text-embedding-ada-002"
                )
            else:
                # Default to HuggingFace
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            
            logger.info(f"Embeddings initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise
    
    def _initialize_llm(self):
        """Initialize LLM based on organization configuration."""
        try:
            provider = self.organization.llm_provider
            model_name = self.organization.llm_model_name
            
            if provider == 'aws_bedrock':
                self.llm = ChatBedrockConverse(
                    model_id=model_name or "anthropic.claude-3-sonnet-20240229-v1:0",
                    region_name=self.organization.aws_region,
                    temperature=self.organization.temperature,
                    max_tokens=self.organization.max_tokens
                )
            elif provider == 'databricks':
                self.llm = ChatDatabricks(
                    endpoint=f"{self.organization.databricks_host}/serving-endpoints/{model_name}",
                    databricks_token=self.organization.databricks_token,
                    temperature=self.organization.temperature,
                    max_tokens=self.organization.max_tokens
                )
            elif provider == 'openai':
                self.llm = ChatOpenAI(
                    model_name=model_name or "gpt-3.5-turbo",
                    temperature=self.organization.temperature,
                    max_tokens=self.organization.max_tokens
                )
            else:
                # Default to HuggingFace
                hf_llm = HuggingFaceEndpoint(
                    repo_id=model_name,
                    task="text-generation",
                    max_new_tokens=self.organization.max_tokens,
                    temperature=self.organization.temperature,
                    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
                )
                self.llm = ChatHuggingFace(llm=hf_llm, verbose=True)
            
            logger.info(f"LLM initialized: {provider} - {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_reranker(self):
        """Initialize reranker for improved retrieval."""
        if self.organization.enable_reranking:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Reranker initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {str(e)}")
                self.reranker = None
    
    def _initialize_monitoring(self):
        """Initialize monitoring and tracing."""
        try:
            if hasattr(settings, 'LANGSMITH_API_KEY'):
                self.langsmith_client = LangSmithClient(
                    api_key=settings.LANGSMITH_API_KEY
                )
                logger.info("LangSmith monitoring initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize monitoring: {str(e)}")
    
    def create_knowledge_map_in_pinecone(
        self, 
        documents: List[Document], 
        knowledge_map_name: str,
        tags: Optional[List[str]] = None
    ) -> KnowledgeMap:
        """Create a knowledge map by processing documents and storing in Pinecone."""
        
        start_time = time.time()
        
        # Create knowledge map record
        knowledge_map = KnowledgeMap.objects.create(
            organization=self.organization,
            name=knowledge_map_name,
            description=f"Knowledge map created from {len(documents)} documents"
        )
        
        # Process each document
        total_chunks = 0
        for document in documents:
            try:
                chunks_created = self._process_document_to_vectors(
                    document, 
                    knowledge_map, 
                    tags or []
                )
                total_chunks += chunks_created
                
                # Add document to knowledge map
                knowledge_map.documents.add(document)
                
            except Exception as e:
                logger.error(f"Failed to process document {document.filename}: {str(e)}")
                continue
        
        # Update knowledge map statistics
        knowledge_map.total_entities = total_chunks
        knowledge_map.save()
        
        processing_time = time.time() - start_time
        logger.info(f"Knowledge map '{knowledge_map_name}' created with {total_chunks} chunks in {processing_time:.2f}s")
        
        return knowledge_map
    
    def _process_document_to_vectors(
        self, 
        document: Document, 
        knowledge_map: KnowledgeMap,
        tags: List[str]
    ) -> int:
        """Process a single document into vector chunks."""
        
        # Get document text
        full_text = self._extract_document_text(document)
        if not full_text:
            return 0
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.organization.chunk_size,
            chunk_overlap=self.organization.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_text(full_text)
        
        # Process chunks in batches
        batch_size = 100
        chunks_created = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            chunks_created += self._add_chunks_to_pinecone(
                batch, document, knowledge_map, tags, i
            )
        
        return chunks_created
    
    def _extract_document_text(self, document: Document) -> str:
        """Extract text from document based on type."""
        from .utils import DocumentProcessor
        
        try:
            if document.document_type == 'pdf':
                chunks = DocumentProcessor.process_pdf(document.file.path)
            elif document.document_type == 'excel':
                chunks = DocumentProcessor.process_excel(document.file.path)
            elif document.document_type == 'word':
                chunks = DocumentProcessor.process_word(document.file.path)
            else:
                return ""
            
            return " ".join([chunk["text"] for chunk in chunks])
            
        except Exception as e:
            logger.error(f"Failed to extract text from {document.filename}: {str(e)}")
            return ""
    
    def _add_chunks_to_pinecone(
        self, 
        chunks: List[str], 
        document: Document,
        knowledge_map: KnowledgeMap,
        tags: List[str],
        start_index: int
    ) -> int:
        """Add chunks to Pinecone vector store."""
        
        if not self.vector_index or not self.embeddings:
            return 0
        
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(chunks)
            
            # Prepare vectors for upsert
            vectors = []
            vector_chunks = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_index = start_index + i
                vector_id = f"{document.id}_{chunk_index}_{uuid.uuid4().hex[:8]}"
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()
                
                # Create metadata
                metadata = {
                    "text": chunk,
                    "document_id": str(document.id),
                    "document_name": document.filename,
                    "document_type": document.document_type,
                    "organization_id": str(self.organization.id),
                    "knowledge_map_id": str(knowledge_map.id),
                    "chunk_index": chunk_index,
                    "tags": tags,
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Create VectorChunk record
                vector_chunk = VectorChunk(
                    organization=self.organization,
                    document=document,
                    vector_index=VectorIndex.objects.get(
                        organization=self.organization,
                        name=self.organization.pinecone_index_name or f"org-{self.organization.slug}"
                    ),
                    vector_id=vector_id,
                    embedding_model=self.organization.embedding_model_name,
                    embedding_dimension=len(embedding),
                    chunk_text=chunk,
                    chunk_hash=chunk_hash,
                    chunk_index=chunk_index,
                    tags=tags,
                    processing_metadata={
                        "knowledge_map_id": str(knowledge_map.id),
                        "processing_time": datetime.now(timezone.utc).isoformat()
                    }
                )
                vector_chunks.append(vector_chunk)
            
            # Upsert to Pinecone
            self.vector_index.upsert(
                vectors=vectors,
                namespace=self.organization.pinecone_namespace
            )
            
            # Bulk create VectorChunk records
            VectorChunk.objects.bulk_create(vector_chunks)
            
            logger.info(f"Added {len(vectors)} chunks to Pinecone for document {document.filename}")
            return len(vectors)
            
        except Exception as e:
            logger.error(f"Failed to add chunks to Pinecone: {str(e)}")
            return 0
    
    def update_pinecone_tags(self, vector_ids: List[str], new_tags: List[str]):
        """Update tags for specific vectors in Pinecone."""
        try:
            # Update in Pinecone
            for vector_id in vector_ids:
                self.vector_index.update(
                    id=vector_id,
                    set_metadata={"tags": new_tags},
                    namespace=self.organization.pinecone_namespace
                )
            
            # Update in database
            VectorChunk.objects.filter(
                organization=self.organization,
                vector_id__in=vector_ids
            ).update(tags=new_tags)
            
            logger.info(f"Updated tags for {len(vector_ids)} vectors")
            
        except Exception as e:
            logger.error(f"Failed to update tags: {str(e)}")
    
    def delete_entity(self, entity_ids: List[str]):
        """Delete entities (vectors) from Pinecone and database."""
        try:
            # Delete from Pinecone
            self.vector_index.delete(
                ids=entity_ids,
                namespace=self.organization.pinecone_namespace
            )
            
            # Delete from database
            VectorChunk.objects.filter(
                organization=self.organization,
                vector_id__in=entity_ids
            ).delete()
            
            logger.info(f"Deleted {len(entity_ids)} entities")
            
        except Exception as e:
            logger.error(f"Failed to delete entities: {str(e)}")
    
    def advanced_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        enable_reranking: Optional[bool] = None,
        enable_multi_query: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Perform advanced query with multi-query generation, reranking, and confidence scoring."""
        
        start_time = time.time()
        
        # Get or create session
        if session_id:
            session, _ = QuerySession.objects.get_or_create(
                organization=self.organization,
                session_id=session_id,
                defaults={'user_identifier': 'anonymous'}
            )
        else:
            session = None
        
        # Use organization settings if not specified
        if enable_reranking is None:
            enable_reranking = self.organization.enable_reranking
        if enable_multi_query is None:
            enable_multi_query = self.organization.enable_multi_query
        
        try:
            # Generate multiple query variations if enabled
            queries = [query]
            if enable_multi_query and self.llm:
                queries = self._generate_multi_queries(query)
            
            # Retrieve documents for each query
            all_docs = []
            for q in queries:
                docs = self._semantic_search(q, filters, top_k * 2)  # Get more for reranking
                all_docs.extend(docs)
            
            # Remove duplicates
            unique_docs = self._deduplicate_documents(all_docs)
            
            # Rerank if enabled
            if enable_reranking and self.reranker and len(unique_docs) > 1:
                unique_docs = self._rerank_documents(query, unique_docs)
            
            # Limit to top_k
            final_docs = unique_docs[:top_k]
            
            # Generate response
            response = self._generate_response(query, final_docs)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(query, final_docs, response)
            
            # Detect obligations
            has_obligations = self._detect_obligations(response) if self.organization.enable_obligation_detection else False
            
            # Log query
            processing_time = int((time.time() - start_time) * 1000)
            if session:
                QueryLog.objects.create(
                    organization=self.organization,
                    session=session,
                    original_query=query,
                    processed_queries=queries,
                    response_text=response,
                    confidence_score=confidence_score,
                    has_obligations=has_obligations,
                    retrieved_chunks=[{
                        "vector_id": doc.metadata.get("vector_id"),
                        "text": doc.page_content[:200],
                        "score": doc.metadata.get("score", 0)
                    } for doc in final_docs],
                    retrieval_method="advanced_semantic",
                    reranking_applied=enable_reranking,
                    processing_time_ms=processing_time
                )
            
            return {
                "response": response,
                "confidence_score": confidence_score,
                "has_obligations": has_obligations,
                "sources": [{
                    "document_name": doc.metadata.get("document_name"),
                    "chunk_text": doc.page_content[:300],
                    "score": doc.metadata.get("score", 0),
                    "vector_id": doc.metadata.get("vector_id")
                } for doc in final_docs],
                "processing_time_ms": processing_time,
                "queries_used": queries,
                "reranking_applied": enable_reranking
            }
            
        except Exception as e:
            logger.error(f"Advanced query failed: {str(e)}")
            raise
    
    def _generate_multi_queries(self, original_query: str) -> List[str]:
        """Generate multiple query variations for better retrieval."""
        try:
            prompt_template = """You are an AI language model assistant. Your task is to generate 3 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help 
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            
            Original question: {question}"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["question"]
            )
            
            llm_chain = LLMChain(llm=self.llm, prompt=prompt, output_parser=LineListOutputParser())
            alternative_queries = llm_chain.run(question=original_query)
            
            # Include original query
            all_queries = [original_query] + alternative_queries
            return all_queries[:4]  # Limit to 4 queries total
            
        except Exception as e:
            logger.warning(f"Failed to generate multi-queries: {str(e)}")
            return [original_query]
    
    def _semantic_search(
        self, 
        query: str, 
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[LangChainDocument]:
        """Perform semantic search in Pinecone."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build filter
            pinecone_filter = {"organization_id": str(self.organization.id)}
            if filters:
                pinecone_filter.update(filters)
            
            # Search in Pinecone
            results = self.vector_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter,
                namespace=self.organization.pinecone_namespace
            )
            
            # Convert to LangChain documents
            documents = []
            for match in results.matches:
                doc = LangChainDocument(
                    page_content=match.metadata.get("text", ""),
                    metadata={
                        **match.metadata,
                        "score": match.score,
                        "vector_id": match.id
                    }
                )
                documents.append(doc)
            
            # Update retrieval statistics
            vector_ids = [match.id for match in results.matches]
            VectorChunk.objects.filter(
                organization=self.organization,
                vector_id__in=vector_ids
            ).update(
                retrieval_count=models.F('retrieval_count') + 1,
                last_retrieved=django_timezone.now()
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    def _deduplicate_documents(self, documents: List[LangChainDocument]) -> List[LangChainDocument]:
        """Remove duplicate documents based on content similarity."""
        if not documents:
            return []
        
        unique_docs = []
        seen_hashes = set()
        
        for doc in documents:
            # Create hash of content
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rerank_documents(
        self, 
        query: str, 
        documents: List[LangChainDocument]
    ) -> List[LangChainDocument]:
        """Rerank documents using cross-encoder model."""
        if not self.reranker or len(documents) <= 1:
            return documents
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Sort documents by reranking scores
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Update metadata with reranking scores
            reranked_docs = []
            for doc, score in doc_score_pairs:
                doc.metadata["rerank_score"] = float(score)
                reranked_docs.append(doc)
            
            return reranked_docs
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return documents
    
    def _generate_response(self, query: str, documents: List[LangChainDocument]) -> str:
        """Generate response using LLM and retrieved documents."""
        try:
            # Prepare context
            context = "\n\n".join([
                f"Source {i+1}: {doc.page_content}"
                for i, doc in enumerate(documents)
            ])
            
            # Create prompt
            prompt_template = """You are a helpful AI assistant that answers questions based on the provided context.
            
            Instructions:
            1. Use only the information provided in the context below
            2. If the answer is not in the context, say "I don't have enough information to answer this question"
            3. Provide specific references to sources when possible
            4. Be concise but comprehensive
            5. Format your response in HTML for better readability
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            response = llm_chain.run(context=context, question=query)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."
    
    def _calculate_confidence_score(
        self, 
        query: str, 
        documents: List[LangChainDocument], 
        response: str
    ) -> float:
        """Calculate confidence score for the response."""
        if not documents:
            return 0.0
        
        try:
            # Factors for confidence calculation
            factors = []
            
            # 1. Average similarity score
            scores = [doc.metadata.get("score", 0) for doc in documents]
            avg_score = sum(scores) / len(scores) if scores else 0
            factors.append(avg_score)
            
            # 2. Number of relevant documents
            doc_factor = min(len(documents) / 5.0, 1.0)  # Normalize to 0-1
            factors.append(doc_factor)
            
            # 3. Response length (longer responses might be more comprehensive)
            response_length_factor = min(len(response) / 500.0, 1.0)  # Normalize to 0-1
            factors.append(response_length_factor)
            
            # 4. Reranking scores if available
            rerank_scores = [doc.metadata.get("rerank_score", 0) for doc in documents]
            if any(rerank_scores):
                avg_rerank = sum(rerank_scores) / len(rerank_scores)
                factors.append(avg_rerank)
            
            # Calculate weighted average
            confidence = sum(factors) / len(factors)
            return min(max(confidence, 0.0), 1.0)  # Clamp to 0-1
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def _detect_obligations(self, response: str) -> bool:
        """Detect if the response contains obligations or requirements."""
        obligation_keywords = [
            "must", "shall", "required", "mandatory", "obligation", 
            "compliance", "regulation", "law", "legal", "contractual",
            "binding", "enforce", "penalty", "violation"
        ]
        
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in obligation_keywords)
    
    def get_organization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the organization."""
        try:
            stats = {
                "total_documents": Document.objects.filter(
                    # Assuming documents are linked to organization through some relation
                ).count(),
                "total_vectors": VectorChunk.objects.filter(
                    organization=self.organization
                ).count(),
                "total_queries": QueryLog.objects.filter(
                    organization=self.organization
                ).count(),
                "knowledge_maps": KnowledgeMap.objects.filter(
                    organization=self.organization,
                    is_active=True
                ).count(),
                "vector_indices": VectorIndex.objects.filter(
                    organization=self.organization,
                    is_active=True
                ).count()
            }
            
            # Recent activity
            from django.utils import timezone
            from datetime import timedelta
            
            last_7_days = timezone.now() - timedelta(days=7)
            stats["queries_last_7_days"] = QueryLog.objects.filter(
                organization=self.organization,
                created_at__gte=last_7_days
            ).count()
            
            # Average confidence score
            avg_confidence = QueryLog.objects.filter(
                organization=self.organization,
                confidence_score__isnull=False
            ).aggregate(
                avg_confidence=models.Avg('confidence_score')
            )['avg_confidence']
            
            stats["average_confidence"] = round(avg_confidence or 0, 3)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get organization statistics: {str(e)}")
            return {}

# Utility functions for backward compatibility
def create_vector_service(organization_slug: str) -> AdvancedVectorService:
    """Create vector service instance for organization."""
    try:
        organization = Organization.objects.get(slug=organization_slug, is_active=True)
        return AdvancedVectorService(organization)
    except Organization.DoesNotExist:
        raise ValueError(f"Organization with slug '{organization_slug}' not found")

def get_default_organization() -> Organization:
    """Get or create default organization for backward compatibility."""
    org, created = Organization.objects.get_or_create(
        slug='default',
        defaults={
            'name': 'Default Organization',
            'pinecone_api_key': os.getenv('PINECONE_API_KEY', ''),
            'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', ''),
            'pinecone_index_name': os.getenv('PINECONE_INDEX_NAME', 'methodology-index'),
        }
    )
    return org 