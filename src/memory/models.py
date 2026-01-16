"""
Data models for the Memory Engine.
"""
from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A fact stored in memory."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "default"  # Single user for now
    fact: str
    category: str
    source: str = "conversation"  # "conversation", "explicit", "inferred"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list] = None  # Stored as list for JSON serialization
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class HumanProfile(BaseModel):
    """User profile information."""
    user_id: str = "default"
    name: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


# API Request/Response models
class FactCreate(BaseModel):
    """Request model for creating a fact."""
    fact: str
    category: str
    source: str = "conversation"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FactResponse(BaseModel):
    """Response model for a fact."""
    id: str
    fact: str
    category: str
    source: str
    confidence: float
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class FactSearchRequest(BaseModel):
    """Request model for searching facts."""
    query: str
    category: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

