from django.urls import path
from .views import (
    PDFUploadView,
    DocumentListView,
    ReindexDocumentsView,
    RetrieveContextView,
    AskQuestionView,
    DeleteDocumentView,
    ClearAllDocumentsView,
    DocumentTagsView,
    TagListView,
    TagDetailView
)

urlpatterns = [
    # Document Management
    path('upload-pdf/', PDFUploadView.as_view(), name='upload-pdf'),
    path('documents/', DocumentListView.as_view(), name='list-documents'),
    path('documents/<uuid:document_id>/', DeleteDocumentView.as_view(), name='delete-document'),
    path('documents/<uuid:document_id>/tags/', DocumentTagsView.as_view(), name='update-document-tags'),
    path('documents/clear-all/', ClearAllDocumentsView.as_view(), name='clear-all-documents'),
    
    # Tag Management
    path('tags/', TagListView.as_view(), name='tag-list'),
    path('tags/<int:pk>/', TagDetailView.as_view(), name='tag-detail'),
    
    # Document Processing
    path('reindex/', ReindexDocumentsView.as_view(), name='reindex-documents'),
    
    # Query and Context
    path('retrieve-context/', RetrieveContextView.as_view(), name='retrieve-context'),
    path('ask/', AskQuestionView.as_view(), name='ask-question'),
] 