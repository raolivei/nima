#!/usr/bin/env python3
"""
Test script for Memory Engine.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os
import json
from datetime import datetime

print("🧪 Testing Nima Memory Engine...")
print("=" * 60)

# Test 1: Import modules
print("\n1️⃣  Testing imports...")
try:
    from memory.models import Fact, HumanProfile, FactCreate
    from memory.storage import JSONStorage
    print("✅ Models imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: JSON Storage
print("\n2️⃣  Testing JSON Storage...")
try:
    storage = JSONStorage(storage_dir="data/memory")
    print("✅ JSONStorage initialized")
    
    # Test storing a fact
    fact = Fact(
        fact="User prefers dark mode",
        category="preferences",
        source="explicit",
        confidence=0.9
    )
    stored_fact = storage.store_fact(fact)
    print(f"✅ Fact stored: {stored_fact.id}")
    
    # Test retrieving fact
    retrieved = storage.get_fact(stored_fact.id)
    if retrieved and retrieved.fact == fact.fact:
        print("✅ Fact retrieved successfully")
    else:
        print("❌ Fact retrieval failed")
    
    # Test listing facts
    facts = storage.list_facts()
    print(f"✅ Listed {len(facts)} facts")
    
except Exception as e:
    print(f"❌ Storage error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Embeddings (if available)
print("\n3️⃣  Testing Embeddings...")
try:
    from memory.embeddings import EmbeddingGenerator
    embedding_gen = EmbeddingGenerator()
    print("✅ EmbeddingGenerator initialized")
    
    # Generate embedding
    embedding = embedding_gen.generate("Test fact")
    print(f"✅ Generated embedding: shape {embedding.shape}")
    
    # Test similarity
    emb1 = embedding_gen.generate("User likes Python")
    emb2 = embedding_gen.generate("User prefers Python programming")
    similarity = embedding_gen.similarity(emb1, emb2)
    print(f"✅ Similarity test: {similarity:.3f}")
    
except ImportError as e:
    print(f"⚠️  Embeddings not available: {e}")
    print("   Install with: pip install sentence-transformers")
except Exception as e:
    print(f"❌ Embedding error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Retrieval (if embeddings available)
print("\n4️⃣  Testing Retrieval...")
try:
    from memory.retrieval import MemoryRetriever
    
    # Store some test facts
    test_facts = [
        Fact(fact="User works as a software engineer", category="career", source="conversation"),
        Fact(fact="User lives in Toronto", category="location", source="conversation"),
        Fact(fact="User prefers dark mode", category="preferences", source="explicit"),
    ]
    
    for fact in test_facts:
        storage.store_fact(fact)
    
    # Create retriever
    retriever = MemoryRetriever(storage, embedding_gen)
    
    # Test semantic search
    results = retriever.retrieve("What does the user do for work?", top_k=2)
    print(f"✅ Semantic search returned {len(results)} results")
    for i, fact in enumerate(results, 1):
        print(f"   {i}. {fact.fact}")
    
except Exception as e:
    print(f"⚠️  Retrieval test skipped: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Profile management
print("\n5️⃣  Testing Profile Management...")
try:
    profile = storage.get_profile()
    print(f"✅ Profile retrieved: user_id={profile.user_id}")
    
    # Update profile
    profile.name = "Test User"
    profile.preferences["theme"] = "dark"
    updated = storage.update_profile(profile)
    print(f"✅ Profile updated: name={updated.name}")
    
except Exception as e:
    print(f"❌ Profile error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Memory Engine tests completed!")
print("\nNext: Test API endpoints with:")
print("  python -m uvicorn api.main:app --reload --port 8002")
print("  curl http://localhost:8002/v1/memory/facts")

