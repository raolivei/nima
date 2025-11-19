"""
RAG retrieval logic for memory engine.
"""
from typing import List, Optional
import numpy as np

from .models import Fact
from .storage import MemoryStorage
from .embeddings import EmbeddingGenerator


class MemoryRetriever:
    """Retrieves relevant facts using RAG (Retrieval-Augmented Generation)."""
    
    def __init__(
        self,
        storage: MemoryStorage,
        embedding_generator: EmbeddingGenerator
    ):
        """
        Initialize memory retriever.
        
        Args:
            storage: Storage backend
            embedding_generator: Embedding generator
        """
        self.storage = storage
        self.embedding_generator = embedding_generator
    
    def retrieve(
        self,
        query: str,
        user_id: str = "default",
        category: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.0
    ) -> List[Fact]:
        """
        Retrieve relevant facts using semantic search.
        
        Args:
            query: Search query
            user_id: User ID to filter facts
            category: Optional category filter
            top_k: Number of results to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of relevant facts, sorted by relevance
        """
        # Get all facts (filtered by user and category)
        all_facts = self.storage.list_facts(
            user_id=user_id,
            category=category,
            limit=None  # Get all for semantic search
        )
        
        if not all_facts:
            return []
        
        # Filter by confidence
        facts = [f for f in all_facts if f.confidence >= min_confidence]
        
        if not facts:
            return []
        
        # Generate embeddings for facts that don't have them
        facts_to_embed = []
        fact_indices = []
        for idx, fact in enumerate(facts):
            if fact.embedding is None:
                facts_to_embed.append(fact)
                fact_indices.append(idx)
        
        if facts_to_embed:
            # Generate embeddings
            texts = [f.fact for f in facts_to_embed]
            embeddings = self.embedding_generator.generate_batch(texts)
            
            # Update facts with embeddings
            for i, idx in enumerate(fact_indices):
                facts[idx].embedding = embeddings[i].tolist()
                # Store updated fact with embedding
                self.storage.store_fact(facts[idx])
        
        # Get embeddings for all facts
        embeddings_list = []
        for fact in facts:
            if fact.embedding:
                if isinstance(fact.embedding, list):
                    embedding = np.array(fact.embedding)
                else:
                    embedding = fact.embedding
                embeddings_list.append(embedding)
            else:
                # Fallback: generate on the fly
                embedding = self.embedding_generator.generate(fact.fact)
                embeddings_list.append(embedding)
        
        # Perform semantic search
        query_embedding = self.embedding_generator.generate(query)
        
        similarities = []
        for idx, embedding in enumerate(embeddings_list):
            sim = self.embedding_generator.similarity(query_embedding, embedding)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k facts
        results = []
        for idx, sim in similarities[:top_k]:
            results.append(facts[idx])
        
        return results
    
    def retrieve_by_keywords(
        self,
        keywords: List[str],
        user_id: str = "default",
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[Fact]:
        """
        Retrieve facts by keyword matching (fallback for exact matches).
        
        Args:
            keywords: List of keywords to search for
            user_id: User ID to filter facts
            category: Optional category filter
            limit: Maximum number of results
            
        Returns:
            List of matching facts
        """
        facts = self.storage.list_facts(
            user_id=user_id,
            category=category,
            limit=None
        )
        
        # Simple keyword matching
        keywords_lower = [k.lower() for k in keywords]
        matches = []
        
        for fact in facts:
            fact_lower = fact.fact.lower()
            # Check if any keyword appears in the fact
            if any(keyword in fact_lower for keyword in keywords_lower):
                matches.append(fact)
        
        return matches[:limit]

