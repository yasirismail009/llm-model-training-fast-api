from django.db import models
import uuid
import os

class PDFDocument(models.Model):
    """Model to store information about uploaded PDF documents."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='pdf_uploads/')
    total_pages = models.IntegerField(default=0)
    total_chunks = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.filename

    def delete(self, *args, **kwargs):
        # Delete the file when the model instance is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)

class DocumentChunk(models.Model):
    """Model to store chunks of text from PDF documents."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name='chunks')
    chunk_text = models.TextField()
    page_number = models.IntegerField()
    chunk_index = models.IntegerField()  # Index of chunk within the page
    vector_id = models.CharField(max_length=255, unique=True)  # Pinecone vector ID
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['document', 'page_number', 'chunk_index']

    def __str__(self):
        return f"{self.document.filename} - Page {self.page_number} - Chunk {self.chunk_index}"
