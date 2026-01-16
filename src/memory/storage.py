"""
Storage backends for memory engine.
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np

from .models import Fact, HumanProfile


class MemoryStorage:
    """Base class for memory storage backends."""
    
    def store_fact(self, fact: Fact) -> Fact:
        """Store a fact."""
        raise NotImplementedError
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        raise NotImplementedError
    
    def list_facts(
        self,
        user_id: str = "default",
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Fact]:
        """List facts with optional filtering."""
        raise NotImplementedError
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact."""
        raise NotImplementedError
    
    def get_profile(self, user_id: str = "default") -> HumanProfile:
        """Get user profile."""
        raise NotImplementedError
    
    def update_profile(self, profile: HumanProfile) -> HumanProfile:
        """Update user profile."""
        raise NotImplementedError


class JSONStorage(MemoryStorage):
    """JSON file-based storage (for development)."""
    
    def __init__(self, storage_dir: str = "data/memory"):
        """
        Initialize JSON storage.
        
        Args:
            storage_dir: Directory to store JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.facts_file = self.storage_dir / "facts.json"
        self.profile_file = self.storage_dir / "profile.json"
        
        # Initialize files if they don't exist
        if not self.facts_file.exists():
            self._save_facts({})
        if not self.profile_file.exists():
            self._save_profile(HumanProfile())
    
    def _load_facts(self) -> Dict[str, dict]:
        """Load facts from JSON file."""
        if not self.facts_file.exists():
            return {}
        
        with open(self.facts_file, 'r') as f:
            data = json.load(f)
            return data.get('facts', {})
    
    def _save_facts(self, facts: Dict[str, dict]):
        """Save facts to JSON file."""
        with open(self.facts_file, 'w') as f:
            json.dump({'facts': facts}, f, indent=2, default=str)
    
    def _load_profile(self) -> HumanProfile:
        """Load profile from JSON file."""
        if not self.profile_file.exists():
            return HumanProfile()
        
        with open(self.profile_file, 'r') as f:
            data = json.load(f)
            return HumanProfile(**data)
    
    def _save_profile(self, profile: HumanProfile):
        """Save profile to JSON file."""
        with open(self.profile_file, 'w') as f:
            json.dump(profile.dict(), f, indent=2, default=str)
    
    def store_fact(self, fact: Fact) -> Fact:
        """Store a fact."""
        facts = self._load_facts()
        
        # Convert embedding to list if present
        fact_dict = fact.dict()
        if fact_dict.get('embedding') is not None:
            if isinstance(fact_dict['embedding'], np.ndarray):
                fact_dict['embedding'] = fact_dict['embedding'].tolist()
        
        # Update timestamps
        if fact.id in facts:
            fact_dict['updated_at'] = datetime.utcnow().isoformat()
        else:
            fact_dict['created_at'] = datetime.utcnow().isoformat()
            fact_dict['updated_at'] = datetime.utcnow().isoformat()
        
        facts[fact.id] = fact_dict
        self._save_facts(facts)
        
        return Fact(**fact_dict)
    
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Get a fact by ID."""
        facts = self._load_facts()
        if fact_id not in facts:
            return None
        
        return Fact(**facts[fact_id])
    
    def list_facts(
        self,
        user_id: str = "default",
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Fact]:
        """List facts with optional filtering."""
        facts = self._load_facts()
        result = []
        
        for fact_data in facts.values():
            fact = Fact(**fact_data)
            
            # Filter by user_id
            if fact.user_id != user_id:
                continue
            
            # Filter by category
            if category and fact.category != category:
                continue
            
            result.append(fact)
        
        # Sort by created_at (newest first)
        result.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            result = result[:limit]
        
        return result
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact."""
        facts = self._load_facts()
        if fact_id not in facts:
            return False
        
        del facts[fact_id]
        self._save_facts(facts)
        return True
    
    def get_profile(self, user_id: str = "default") -> HumanProfile:
        """Get user profile."""
        profile = self._load_profile()
        if profile.user_id != user_id:
            # Create new profile for this user
            profile = HumanProfile(user_id=user_id)
            self._save_profile(profile)
        return profile
    
    def update_profile(self, profile: HumanProfile) -> HumanProfile:
        """Update user profile."""
        profile.updated_at = datetime.utcnow()
        self._save_profile(profile)
        return profile


class PostgreSQLStorage(MemoryStorage):
    """PostgreSQL storage backend (for production)."""
    
    def __init__(self, database_url: str):
        """
        Initialize PostgreSQL storage.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        # TODO: Implement PostgreSQL storage in Phase 1
        # This will require SQLAlchemy models and migrations
        raise NotImplementedError("PostgreSQL storage will be implemented in Phase 1")

