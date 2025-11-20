"""
Database Schemas for Ninety-Nine

Each Pydantic model represents a MongoDB collection. The collection name is the lowercase of the class name.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Chat(BaseModel):
    """Chats collection schema (collection name: "chat")"""
    title: str = Field(..., description="Display title of the chat")
    welcome_shown: bool = Field(True, description="Whether the welcome message has been added")

class Message(BaseModel):
    """Messages collection schema (collection name: "message")"""
    chat_id: str = Field(..., description="Reference to the chat this message belongs to")
    role: str = Field(..., description="Role of the message: user | assistant | system | model | consensus")
    content: str = Field(..., description="Text content of the message")
    model: Optional[str] = Field(None, description="Which model produced it, if applicable: gpt5 | gemini | entropy")
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata, e.g., similarity, file refs")

class ModelResponse(BaseModel):
    """ModelResponses collection schema (collection name: "modelresponse")"""
    chat_id: str = Field(..., description="Chat reference")
    prompt: str = Field(..., description="Original prompt sent to models")
    responses: Dict[str, str] = Field(default_factory=dict, description="Raw responses keyed by model id")
    similarities: Dict[str, float] = Field(default_factory=dict, description="Pairwise similarities, keys: 'gpt5_gemini', 'gpt5_entropy', 'gemini_entropy'")
    consensus: Optional[str] = Field(None, description="Final consensus answer text")
