import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def process_pdf(file_path: str):
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
        raise Exception(f"Failed to process PDF: {str(e)}")
    
    if not chunks_with_metadata:
        raise Exception("No text could be extracted from the PDF. It may be scanned or protected.")
    
    return chunks_with_metadata 