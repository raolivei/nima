"""
FastAPI server for Nima inference service.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from models.transformer import GPTModel
from data.tokenizer import SimpleBPETokenizer, CharTokenizer, WordTokenizer

app = FastAPI(title="Nima API", version="0.5.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None

# In-memory conversation storage (use Redis in production)
conversations: dict = {}

class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.8
    top_k: int = 50

class InferenceResponse(BaseModel):
    response: str
    prompt: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    max_length: int = 300
    temperature: float = 0.8
    top_k: int = 50
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    messages: List[ChatMessage]
    conversation_id: str
def load_model(checkpoint_path: str, tokenizer_instance):
    """Load the trained model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get vocab size from tokenizer
    vocab_size = tokenizer_instance.vocab_size if hasattr(tokenizer_instance, 'vocab_size') else len(tokenizer_instance.vocab)
    
    # Infer architecture from checkpoint
    if 'token_embedding.weight' in checkpoint['model_state_dict']:
        actual_vocab_size = checkpoint['model_state_dict']['token_embedding.weight'].shape[0]
        actual_d_model = checkpoint['model_state_dict']['token_embedding.weight'].shape[1]
        actual_max_seq_len = checkpoint['model_state_dict']['pos_embedding.weight'].shape[0]
    else:
        actual_vocab_size = vocab_size
        actual_d_model = 256
        actual_max_seq_len = 512
    
    config = checkpoint.get('config', {})
    
    if hasattr(config, 'model'):
        model_config = config.model
    elif isinstance(config, dict):
        model_config = config
    else:
        model_config = {}
    
    d_model = getattr(model_config, 'd_model', actual_d_model) if hasattr(model_config, 'd_model') else model_config.get('d_model', actual_d_model)
    n_heads = getattr(model_config, 'n_heads', 4) if hasattr(model_config, 'n_heads') else model_config.get('n_heads', 4)
    n_layers = getattr(model_config, 'n_layers', 4) if hasattr(model_config, 'n_layers') else model_config.get('n_layers', 4)
    d_ff = getattr(model_config, 'd_ff', 1024) if hasattr(model_config, 'd_ff') else model_config.get('d_ff', 1024)
    max_seq_len = getattr(model_config, 'max_seq_length', actual_max_seq_len) if hasattr(model_config, 'max_seq_length') else model_config.get('max_seq_length', actual_max_seq_len)
    dropout = getattr(model_config, 'dropout', 0.1) if hasattr(model_config, 'dropout') else model_config.get('dropout', 0.1)
    
    vocab_size = actual_vocab_size
    
    model_instance = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout
    )
    
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    model_instance.eval()
    
    return model_instance

def generate_text(model_instance, tokenizer_instance, prompt: str, max_length: int = 200, temperature: float = 0.8, top_k: int = 50):
    """Generate text from a prompt."""
    device = next(model_instance.parameters()).device
    
    max_seq_len = model_instance.pos_embedding.weight.shape[0] if hasattr(model_instance, 'pos_embedding') else 512
    
    tokens = tokenizer_instance.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    model_instance.eval()
    with torch.no_grad():
        for _ in range(max_length):
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]
            
            outputs = model_instance(input_ids)
            logits = outputs[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            pad_id = getattr(tokenizer_instance, 'pad_token_id', None) or getattr(tokenizer_instance, 'pad_id', None)
            if pad_id is not None and next_token.item() == pad_id:
                break
    
    generated_ids = input_ids[0].tolist()
    generated_text = tokenizer_instance.decode(generated_ids)
    
    if prompt in generated_text:
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def format_chat_prompt(messages: List[ChatMessage], system_prompt: Optional[str] = None) -> str:
    """
    Format chat messages into a single prompt for the model.
    
    Args:
        messages: List of chat messages
        system_prompt: Optional system prompt for context
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")
    
    # Build conversation context
    for msg in messages:
        role = msg.role.capitalize()
        content = msg.content
        parts.append(f"{role}: {content}\n\n")
    
    # Add assistant prefix for response
    parts.append("Assistant: ")
    
    return "".join(parts)

def generate_streaming(model_instance, tokenizer_instance, prompt: str, max_length: int = 300, temperature: float = 0.8, top_k: int = 50):
    """Generate text with streaming support."""
    device = next(model_instance.parameters()).device
    max_seq_len = model_instance.pos_embedding.weight.shape[0] if hasattr(model_instance, 'pos_embedding') else 512
    
    tokens = tokenizer_instance.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    model_instance.eval()
    generated_tokens = []
    last_decoded_length = 0
    
    with torch.no_grad():
        for _ in range(max_length):
            if input_ids.shape[1] > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]
            
            outputs = model_instance(input_ids)
            logits = outputs[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            # Decode accumulated tokens to get current text
            current_text = tokenizer_instance.decode(generated_tokens)
            
            # Yield only new text (incremental)
            if len(current_text) > last_decoded_length:
                new_text = current_text[last_decoded_length:]
                yield new_text
                last_decoded_length = len(current_text)
            
            # Check for end token
            pad_id = getattr(tokenizer_instance, 'pad_token_id', None) or getattr(tokenizer_instance, 'pad_id', None)
            if pad_id is not None and next_token.item() == pad_id:
                break

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global model, tokenizer
    
    checkpoint_path = os.getenv("NIMA_CHECKPOINT_PATH", "/app/experiments/nima_technical/checkpoint_best.pt")
    tokenizer_path = os.getenv("NIMA_TOKENIZER_PATH", "/app/data/processed/technical_example/tokenizer_bpe.json")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    # Load tokenizer
    if 'bpe' in tokenizer_path.lower():
        tokenizer = SimpleBPETokenizer()
    elif 'word' in tokenizer_path.lower():
        tokenizer = WordTokenizer()
    else:
        tokenizer = CharTokenizer()
    
    tokenizer.load(tokenizer_path)
    
    # Load model
    model = load_model(checkpoint_path, tokenizer)
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """Generate text from a prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        response = generate_text(
            model,
            tokenizer,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        return InferenceResponse(prompt=request.prompt, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with conversation history."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        # Get or create conversation ID
        conversation_id = request.conversation_id or str(uuid4())
        
        # System prompt for application context
        system_prompt = (
            "You are Nima, an AI assistant that understands the following applications:\n"
            "- Canopy: A self-hosted personal finance, investment, and budgeting dashboard\n"
            "- SwimTO: Aggregates and displays indoor community pool drop-in swim schedules for Toronto\n"
            "- US Law Severity Map: Interactive choropleth map showing law severity scores and crime statistics\n\n"
            "Answer questions about these applications helpfully and accurately."
        )
        
        # Format prompt from conversation history
        prompt = format_chat_prompt(request.messages, system_prompt)
        
        # Generate response
        response_text = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        
        # Update conversation history
        updated_messages = request.messages + [
            ChatMessage(role="assistant", content=response_text)
        ]
        
        # Store conversation (limit to last 20 messages)
        conversations[conversation_id] = updated_messages[-20:]
        
        return ChatResponse(
            response=response_text,
            messages=updated_messages,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    
    try:
        conversation_id = request.conversation_id or str(uuid4())
        
        system_prompt = (
            "You are Nima, an AI assistant that understands the following applications:\n"
            "- Canopy: A self-hosted personal finance, investment, and budgeting dashboard\n"
            "- SwimTO: Aggregates and displays indoor community pool drop-in swim schedules for Toronto\n"
            "- US Law Severity Map: Interactive choropleth map showing law severity scores and crime statistics\n\n"
            "Answer questions about these applications helpfully and accurately."
        )
        
        prompt = format_chat_prompt(request.messages, system_prompt)
        
        def generate():
            for text_chunk in generate_streaming(
                model,
                tokenizer,
                prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k
            ):
                if text_chunk:
                    yield f"data: {json.dumps({'content': text_chunk, 'done': False})}\n\n"
            
            # Send final message
            yield f"data: {json.dumps({'content': '', 'done': True, 'conversation_id': conversation_id})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }

@app.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"status": "deleted"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Nima API",
        "version": "0.5.0",
        "endpoints": {
            "health": "/health",
            "inference": "/v1/inference",
            "chat": "/v1/chat",
            "chat_stream": "/v1/chat/stream",
            "conversations": "/v1/conversations/{id}"
        }
    }

