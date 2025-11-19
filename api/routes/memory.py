"""
Memory management API routes.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from memory.models import (
    FactCreate,
    FactResponse,
    FactSearchRequest,
    HumanProfile,
    Fact
)
from memory.storage import JSONStorage, MemoryStorage
from memory.embeddings import EmbeddingGenerator
from memory.retrieval import MemoryRetriever

router = APIRouter(prefix="/v1/memory", tags=["memory"])

# Global storage and retriever instances
_storage: Optional[MemoryStorage] = None
_retriever: Optional[MemoryRetriever] = None


def get_storage() -> MemoryStorage:
    """Get or create storage instance."""
    global _storage
    if _storage is None:
        storage_dir = os.getenv("NIMA_MEMORY_DIR", "data/memory")
        _storage = JSONStorage(storage_dir=storage_dir)
    return _storage


def get_retriever() -> MemoryRetriever:
    """Get or create retriever instance."""
    global _retriever
    if _retriever is None:
        storage = get_storage()
        try:
            embedding_gen = EmbeddingGenerator()
            _retriever = MemoryRetriever(storage, embedding_gen)
        except ImportError:
            # If sentence-transformers not available, create retriever without embeddings
            # Semantic search won't work, but basic storage will
            raise HTTPException(
                status_code=503,
                detail="Embedding model not available. Install sentence-transformers."
            )
    return _retriever


@router.get("/facts", response_model=List[FactResponse])
async def list_facts(
    category: Optional[str] = None,
    limit: int = 50,
    storage: MemoryStorage = Depends(get_storage)
):
    """
    List all facts with optional filtering.
    
    Args:
        category: Optional category filter
        limit: Maximum number of facts to return
        storage: Storage backend
        
    Returns:
        List of facts
    """
    facts = storage.list_facts(
        user_id="default",
        category=category,
        limit=limit
    )
    
    return [
        FactResponse(
            id=fact.id,
            fact=fact.fact,
            category=fact.category,
            source=fact.source,
            confidence=fact.confidence,
            created_at=fact.created_at,
            updated_at=fact.updated_at
        )
        for fact in facts
    ]


@router.post("/facts", response_model=FactResponse, status_code=201)
async def create_fact(
    fact_data: FactCreate,
    storage: MemoryStorage = Depends(get_storage),
    retriever: MemoryRetriever = Depends(get_retriever)
):
    """
    Create a new fact.
    
    Args:
        fact_data: Fact data
        storage: Storage backend
        retriever: Memory retriever (for generating embeddings)
        
    Returns:
        Created fact
    """
    # Create fact
    fact = Fact(
        user_id="default",
        fact=fact_data.fact,
        category=fact_data.category,
        source=fact_data.source,
        confidence=fact_data.confidence,
        metadata=fact_data.metadata
    )
    
    # Generate embedding
    try:
        embedding = retriever.embedding_generator.generate(fact.fact)
        fact.embedding = embedding.tolist()
    except Exception as e:
        # Continue without embedding if generation fails
        print(f"Warning: Could not generate embedding: {e}")
    
    # Store fact
    stored_fact = storage.store_fact(fact)
    
    return FactResponse(
        id=stored_fact.id,
        fact=stored_fact.fact,
        category=stored_fact.category,
        source=stored_fact.source,
        confidence=stored_fact.confidence,
        created_at=stored_fact.created_at,
        updated_at=stored_fact.updated_at
    )


@router.get("/facts/{fact_id}", response_model=FactResponse)
async def get_fact(
    fact_id: str,
    storage: MemoryStorage = Depends(get_storage)
):
    """
    Get a fact by ID.
    
    Args:
        fact_id: Fact ID
        storage: Storage backend
        
    Returns:
        Fact
    """
    fact = storage.get_fact(fact_id)
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    
    return FactResponse(
        id=fact.id,
        fact=fact.fact,
        category=fact.category,
        source=fact.source,
        confidence=fact.confidence,
        created_at=fact.created_at,
        updated_at=fact.updated_at
    )


@router.delete("/facts/{fact_id}", status_code=204)
async def delete_fact(
    fact_id: str,
    storage: MemoryStorage = Depends(get_storage)
):
    """
    Delete a fact.
    
    Args:
        fact_id: Fact ID
        storage: Storage backend
    """
    deleted = storage.delete_fact(fact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fact not found")


@router.post("/search", response_model=List[FactResponse])
async def search_facts(
    search_request: FactSearchRequest,
    retriever: MemoryRetriever = Depends(get_retriever)
):
    """
    Search facts using semantic search.
    
    Args:
        search_request: Search request
        retriever: Memory retriever
        
    Returns:
        List of relevant facts
    """
    facts = retriever.retrieve(
        query=search_request.query,
        user_id="default",
        category=search_request.category,
        top_k=search_request.limit,
        min_confidence=search_request.min_confidence
    )
    
    return [
        FactResponse(
            id=fact.id,
            fact=fact.fact,
            category=fact.category,
            source=fact.source,
            confidence=fact.confidence,
            created_at=fact.created_at,
            updated_at=fact.updated_at
        )
        for fact in facts
    ]


@router.get("/profile", response_model=HumanProfile)
async def get_profile(
    storage: MemoryStorage = Depends(get_storage)
):
    """
    Get user profile.
    
    Args:
        storage: Storage backend
        
    Returns:
        User profile
    """
    return storage.get_profile(user_id="default")


@router.put("/profile", response_model=HumanProfile)
async def update_profile(
    profile: HumanProfile,
    storage: MemoryStorage = Depends(get_storage)
):
    """
    Update user profile.
    
    Args:
        profile: Profile data
        storage: Storage backend
        
    Returns:
        Updated profile
    """
    profile.user_id = "default"
    return storage.update_profile(profile)

