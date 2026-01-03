"""
RAG Engine Module

This module implements the RAG pipeline with Google Gemini LLM.
Uses LangChain Expression Language (LCEL) for modern compatibility.
"""

import logging
import time
from typing import Dict, Any, Optional, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are an expert assistant on data mining and big data analytics for SMEs in the transportation sector.

Use the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information in the knowledge base to answer that question."

Context:
{context}

Question: {question}

Provide a clear, concise, and accurate answer based solely on the context provided:"""


def initialize_gemini(api_key: str, temperature: float = 0.3, model: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """
    Initialize Google Gemini LLM.
    
    Args:
        api_key: Google Gemini API key
        temperature: Model temperature (0-1), lower = more consistent
        model: Model name (default: gemini-2.5-flash, fallback: gemini-2.5-flash-lite)
        
    Returns:
        ChatGoogleGenerativeAI instance
        
    Raises:
        ValueError: If API key is empty
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key is required")
    
    logger.info(f"Initializing Google Gemini ({model})...")
    
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
    )
    
    logger.info(f"Gemini ({model}) initialized successfully")
    return llm


def format_docs(docs: List) -> str:
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG Chain wrapper that stores retriever for source document access."""
    
    def __init__(self, chain, retriever, k: int = 3):
        self.chain = chain
        self.retriever = retriever
        self.k = k
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the RAG chain and return results with source documents."""
        query = inputs.get("query", "")
        
        # Get source documents
        source_docs = self.retriever.invoke(query)
        
        # Get answer from chain
        result = self.chain.invoke(query)
        
        return {
            "result": result,
            "source_documents": source_docs
        }


def create_rag_chain(
    vector_store: FAISS,
    llm: ChatGoogleGenerativeAI,
    k: int = 3
) -> RAGChain:
    """
    Create RAG chain with retriever and LLM using LCEL.
    
    Args:
        vector_store: FAISS vector store instance
        llm: Gemini LLM instance
        k: Number of documents to retrieve (default: 3)
        
    Returns:
        RAGChain instance
    """
    logger.info(f"Creating RAG chain with k={k}...")
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Create LCEL chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain created successfully")
    return RAGChain(chain, retriever, k)


def get_rag_response(
    query: str,
    rag_chain: RAGChain,
    max_retries: int = 1,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Get response from RAG chain for a given query.
    No retry for rate limit errors - let caller handle model switching.
    
    Args:
        query: User's question
        rag_chain: RAG chain instance
        max_retries: Maximum number of retry attempts (default 1 = no retry)
        timeout: Timeout in seconds (not directly used, but for reference)
        
    Returns:
        Dictionary with 'result' and 'source_documents' keys
        
    Raises:
        Exception: For rate limit errors (429) to allow caller to switch models
    """
    if not query or not query.strip():
        return {
            "result": "Please provide a valid question.",
            "source_documents": []
        }
    
    logger.info(f"Processing query: {query[:100]}...")
    
    try:
        start_time = time.time()
        
        # Execute query
        response = rag_chain.invoke({"query": query})
        
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f}s")
        
        # Log retrieved documents
        source_docs = response.get("source_documents", [])
        logger.info(f"Retrieved {len(source_docs)} source documents")
        
        return {
            "result": response.get("result", "No response generated"),
            "source_documents": source_docs
        }
        
    except Exception as e:
        error_str = str(e).lower()
        # Immediately raise rate limit errors for model switching
        if "429" in str(e) or "rate limit" in error_str or "quota" in error_str or "resource" in error_str:
            logger.warning(f"Rate limit error detected, raising for model switch: {str(e)}")
            raise  # Let app.py handle model switching
        
        logger.error(f"Query failed: {str(e)}")
        return {
            "result": f"Error processing your question: {str(e)}",
            "source_documents": []
        }


def format_source_documents(source_documents: list) -> str:
    """
    Format source documents for display.
    
    Args:
        source_documents: List of Document objects
        
    Returns:
        Formatted string of sources
    """
    if not source_documents:
        return "No sources available."
    
    formatted = []
    for i, doc in enumerate(source_documents, 1):
        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        formatted.append(f"**Source {i}:**\n{content_preview}")
    
    return "\n\n".join(formatted)


if __name__ == "__main__":
    # This module requires an API key to test
    print("RAG Engine module loaded successfully")
    print("Use initialize_gemini(api_key) to create LLM instance")
    print("Use create_rag_chain(vector_store, llm) to create RAG chain")
