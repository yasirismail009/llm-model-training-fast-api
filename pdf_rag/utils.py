import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF and split into chunks with accurate page metadata using pdfplumber."""
    chunks_with_metadata = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text with layout preservation
                text = page.extract_text(layout=True)
                if not text or not text.strip():
                    continue
                
                # Extract tables if present
                tables = page.extract_tables()
                table_text = ""
                if tables:
                    for table in tables:
                        # Clean and format table data
                        cleaned_table = []
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            cleaned_table.append(cleaned_row)
                        table_text += "\n".join(["\t".join(row) for row in cleaned_table]) + "\n\n"
                
                # Combine text and tables
                page_text = text + "\n" + table_text if table_text else text
                
                # Add page number and total pages
                page_header = f"[Page {page_num+1} of {total_pages}]\n\n"
                page_text_with_header = page_header + page_text
                
                # Split into chunks
                page_chunks = text_splitter.split_text(page_text_with_header)
                
                for i, chunk in enumerate(page_chunks):
                    chunks_with_metadata.append({
                        "text": chunk,
                        "page_num": page_num + 1,
                        "total_pages": total_pages,
                        "chunk_on_page": i + 1,
                        "has_tables": bool(table_text),
                        "metadata": {
                            "extraction_method": "pdfplumber",
                            "layout_preserved": True
                        }
                    })
                    
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")
    
    if not chunks_with_metadata:
        raise Exception("No text could be extracted from the PDF. It may be scanned or protected.")
    
    return chunks_with_metadata 