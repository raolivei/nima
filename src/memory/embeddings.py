"""
Embedding generation for semantic search.
"""
import os
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingGenerator:
    """Generates embeddings for facts using sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully")
    
    def generate(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            NumPy array of embeddings
        """
        if self.model is None:
            self._load_model()
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            NumPy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if self.model is None:
            self._load_model()
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Normalize to [0, 1] range
        return (similarity + 1) / 2
    
    def search(
        self,
        query: str,
        embeddings: List[np.ndarray],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Search for most similar embeddings.
        
        Args:
            query: Query text
            embeddings: List of embeddings to search through
            top_k: Number of results to return
            
        Returns:
            List of tuples (index, similarity_score)
        """
        query_embedding = self.generate(query)
        
        similarities = []
        for idx, embedding in enumerate(embeddings):
            sim = self.similarity(query_embedding, embedding)
            similarities.append((idx, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

