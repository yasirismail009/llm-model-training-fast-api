from rest_framework import serializers
from .models import PDFDocument, DocumentChunk

class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_text', 'page_number', 'chunk_index', 'vector_id', 'created_at']

class PDFDocumentSerializer(serializers.ModelSerializer):
    chunks = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = PDFDocument
        fields = ['id', 'filename', 'file', 'total_pages', 'total_chunks', 
                 'created_at', 'updated_at', 'chunks']
        read_only_fields = ['total_pages', 'total_chunks', 'created_at', 'updated_at']

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    
class ContextRetrievalSerializer(serializers.Serializer):
    question = serializers.CharField()
    document_source = serializers.CharField(required=False, allow_null=True)
    k = serializers.IntegerField(default=20) 