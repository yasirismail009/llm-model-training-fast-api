from rest_framework import serializers
from .models import PDFDocument, DocumentChunk, Tag

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name', 'created_at', 'updated_at']

class DocumentChunkSerializer(serializers.ModelSerializer):
    class Meta:
        model = DocumentChunk
        fields = ['id', 'chunk_text', 'page_number', 'chunk_index', 'vector_id', 'created_at']

class PDFDocumentSerializer(serializers.ModelSerializer):
    chunks = DocumentChunkSerializer(many=True, read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    tag_names = serializers.ListField(
        child=serializers.CharField(max_length=100),
        write_only=True,
        required=False
    )
    
    class Meta:
        model = PDFDocument
        fields = ['id', 'filename', 'file', 'total_pages', 'total_chunks', 
                 'created_at', 'updated_at', 'chunks', 'tags', 'tag_names']
        read_only_fields = ['total_pages', 'total_chunks', 'created_at', 'updated_at']

    def create(self, validated_data):
        tag_names = validated_data.pop('tag_names', [])
        document = super().create(validated_data)
        
        # Handle tags
        for tag_name in tag_names:
            tag, _ = Tag.objects.get_or_create(name=tag_name)
            document.tags.add(tag)
        
        return document

    def update(self, instance, validated_data):
        tag_names = validated_data.pop('tag_names', None)
        document = super().update(instance, validated_data)
        
        # Update tags if provided
        if tag_names is not None:
            document.tags.clear()
            for tag_name in tag_names:
                tag, _ = Tag.objects.get_or_create(name=tag_name)
                document.tags.add(tag)
        
        return document

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    tags = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        default=list
    )

class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Filter documents by tags"
    )

class ContextRetrievalSerializer(serializers.Serializer):
    question = serializers.CharField()
    document_source = serializers.CharField(required=False, allow_null=True)
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        help_text="Filter documents by tags"
    )
    k = serializers.IntegerField(default=20) 