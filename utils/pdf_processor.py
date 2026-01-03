"""
PDF Processor Module

This module handles PDF text extraction and text chunking for RAG.
"""

import logging
from typing import List
from pathlib import Path

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF is corrupted or cannot be read
    """
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        text_content = []
        
        logger.info(f"Processing PDF with {len(reader.pages)} pages")
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)
            else:
                logger.warning(f"Page {page_num + 1} has no extractable text")
        
        full_text = "\n\n".join(text_content)
        
        if not full_text.strip():
            logger.warning("PDF contains no extractable text (may contain only images)")
            return ""
        
        logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
        return full_text
        
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise Exception(f"Failed to read PDF file: {str(e)}")


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    logger.info(f"Split text into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        print(f"Extracted {len(chunks)} chunks")
        if chunks:
            print(f"First chunk preview: {chunks[0][:200]}...")
