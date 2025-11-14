"""
FastAPI server for Nima inference service.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from models.transformer import GPTModel
from data.tokenizer import SimpleBPETokenizer, CharTokenizer, WordTokenizer

app = FastAPI(title="Nima API", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None

class InferenceRequest(BaseModel):
    prompt: str
    max_length: int = 200
    temperature: float = 0.8
    top_k: int = 50

class InferenceResponse(BaseModel):
    response: str
    prompt: str

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

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Nima API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "inference": "/v1/inference"
        }
    }

