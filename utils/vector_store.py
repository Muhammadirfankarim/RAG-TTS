"""
Vector Store Module

This module manages FAISS vector store for semantic search.
"""

import logging
import os
from typing import List, Optional
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default paths
VECTORSTORE_DIR = "vectorstore"
INDEX_NAME = "faiss_index"


def initialize_embeddings() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings model.
    
    Returns:
        HuggingFaceEmbeddings instance with sentence-transformers model
    """
    logger.info("Initializing HuggingFace embeddings model...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    logger.info("Embeddings model initialized successfully")
    return embeddings


def create_vector_store(
    text_chunks: List[str],
    embeddings_model: HuggingFaceEmbeddings,
    save_path: str = VECTORSTORE_DIR
) -> FAISS:
    """
    Create FAISS vector store from text chunks.
    
    Args:
        text_chunks: List of text chunks to embed
        embeddings_model: HuggingFace embeddings instance
        save_path: Directory to save the vector store
        
    Returns:
        FAISS vector store instance
    """
    if not text_chunks:
        raise ValueError("No text chunks provided for vector store creation")
    
    logger.info(f"Creating vector store from {len(text_chunks)} chunks...")
    
    # Create vector store
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings_model
    )
    
    logger.info("Vector store created successfully")
    
    # Save to disk
    save_vector_store(vector_store, save_path)
    
    return vector_store


def load_vector_store(
    embeddings_model: HuggingFaceEmbeddings,
    load_path: str = VECTORSTORE_DIR
) -> Optional[FAISS]:
    """
    Load existing FAISS vector store from disk.
    
    Args:
        embeddings_model: HuggingFace embeddings instance
        load_path: Directory containing the vector store
        
    Returns:
        FAISS vector store instance, or None if not found
    """
    index_path = Path(load_path) / f"{INDEX_NAME}.faiss"
    
    if not index_path.exists():
        logger.warning(f"Vector store not found at {load_path}")
        return None
    
    try:
        logger.info(f"Loading vector store from {load_path}...")
        
        vector_store = FAISS.load_local(
            folder_path=load_path,
            embeddings=embeddings_model,
            index_name=INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        
        logger.info("Vector store loaded successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None


def save_vector_store(
    vector_store: FAISS,
    save_path: str = VECTORSTORE_DIR
) -> None:
    """
    Save FAISS vector store to disk.
    
    Args:
        vector_store: FAISS vector store instance
        save_path: Directory to save the vector store
    """
    # Create directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    try:
        vector_store.save_local(
            folder_path=save_path,
            index_name=INDEX_NAME
        )
        logger.info(f"Vector store saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        raise


def get_vector_store_info(load_path: str = VECTORSTORE_DIR) -> dict:
    """
    Get information about the vector store.
    
    Args:
        load_path: Directory containing the vector store
        
    Returns:
        Dictionary with vector store information
    """
    index_path = Path(load_path) / f"{INDEX_NAME}.faiss"
    pkl_path = Path(load_path) / f"{INDEX_NAME}.pkl"
    
    info = {
        "exists": index_path.exists() and pkl_path.exists(),
        "index_path": str(index_path),
        "pkl_path": str(pkl_path),
        "index_size": index_path.stat().st_size if index_path.exists() else 0,
        "pkl_size": pkl_path.stat().st_size if pkl_path.exists() else 0,
    }
    
    return info


if __name__ == "__main__":
    # Test the module
    embeddings = initialize_embeddings()
    
    # Test with sample texts
    sample_texts = [
        "Big data analytics helps organizations process large datasets.",
        "Machine learning models can identify patterns in data.",
        "SMEs face challenges in adopting big data technologies."
    ]
    
    vector_store = create_vector_store(sample_texts, embeddings, "test_vectorstore")
    
    # Test similarity search
    results = vector_store.similarity_search("What is big data?", k=2)
    print("Search results:")
    for doc in results:
        print(f"  - {doc.page_content[:100]}...")
