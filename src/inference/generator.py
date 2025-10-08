"""
Inference engine for text generation with language models.

This module provides tools for generating text using trained models with various
sampling strategies and decoding methods.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Callable, Dict, Any, Tuple
import numpy as np


class TextGenerator:
    """
    Text generator for language models with multiple sampling strategies.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize text generator.
        
        Args:
            model: Trained language model
            tokenizer: Tokenizer instance
            device: Device to run on (defaults to cuda if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None
    ) -> List[str]:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p (nucleus sampling)
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to sample (True) or use greedy decoding (False)
            early_stopping: Stop generation at EOS token
            eos_token_id: End-of-sequence token ID
            
        Returns:
            List of generated text strings
        """
        # Encode prompt
        if prompt:
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt, add_special_tokens=True),
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
        else:
            # Start with SOS token if available
            sos_token = self.tokenizer.vocab.get('<sos>', 0)
            input_ids = torch.tensor([[sos_token]], dtype=torch.long).to(self.device)
        
        # Repeat input for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Get EOS token ID
        if eos_token_id is None:
            eos_token_id = self.tokenizer.vocab.get('<eos>', -1)
        
        # Track which sequences have finished
        finished = torch.zeros(num_return_sequences, dtype=torch.bool, device=self.device)
        
        # Generate tokens
        for _ in range(max_length):
            # Forward pass
            logits = self.model(input_ids)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(num_return_sequences):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k is not None:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)
                
                # Apply top-p filtering
                if top_p is not None:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Mark finished sequences
            if early_stopping and eos_token_id >= 0:
                finished |= (next_tokens.squeeze(-1) == eos_token_id)
                if finished.all():
                    break
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        
        # Decode generated sequences
        generated_texts = []
        for i in range(num_return_sequences):
            tokens = input_ids[i].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Filter logits to keep only top-k tokens.
        
        Args:
            logits: Token logits
            top_k: Number of top tokens to keep
            
        Returns:
            Filtered logits
        """
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Filter logits using nucleus (top-p) sampling.
        
        Args:
            logits: Token logits
            top_p: Cumulative probability threshold
            
        Returns:
            Filtered logits
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def beam_search(
        self,
        prompt: str = "",
        max_length: int = 100,
        beam_size: int = 5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_token_id: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Generate text using beam search.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length to generate
            beam_size: Number of beams to keep
            length_penalty: Length penalty (>1.0 favors longer sequences)
            early_stopping: Stop when all beams have generated EOS
            eos_token_id: End-of-sequence token ID
            
        Returns:
            List of (text, score) tuples sorted by score
        """
        # Encode prompt
        if prompt:
            input_ids = torch.tensor(
                self.tokenizer.encode(prompt, add_special_tokens=True),
                dtype=torch.long
            ).unsqueeze(0).to(self.device)
        else:
            sos_token = self.tokenizer.vocab.get('<sos>', 0)
            input_ids = torch.tensor([[sos_token]], dtype=torch.long).to(self.device)
        
        # Get EOS token ID
        if eos_token_id is None:
            eos_token_id = self.tokenizer.vocab.get('<eos>', -1)
        
        # Initialize beams: (sequence, score, finished)
        beams = [(input_ids[0], 0.0, False)]
        finished_beams = []
        
        for step in range(max_length):
            candidates = []
            
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, True))
                    continue
                
                # Forward pass
                with torch.no_grad():
                    logits = self.model(seq.unsqueeze(0))
                    next_token_logits = logits[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k candidates
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for log_prob, token_id in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                    new_score = score + log_prob.item()
                    
                    # Apply length penalty
                    length_normalized_score = new_score / (len(new_seq) ** length_penalty)
                    
                    is_finished = (token_id.item() == eos_token_id)
                    candidates.append((new_seq, length_normalized_score, is_finished))
            
            # Keep top beam_size beams
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Move finished beams to finished list
            if early_stopping:
                new_beams = []
                for seq, score, finished in beams:
                    if finished:
                        finished_beams.append((seq, score))
                    else:
                        new_beams.append((seq, score, finished))
                
                beams = new_beams
                
                if not beams:  # All beams finished
                    break
        
        # Add remaining beams to finished
        for seq, score, _ in beams:
            finished_beams.append((seq, score))
        
        # Decode and return
        results = []
        for seq, score in sorted(finished_beams, key=lambda x: x[1], reverse=True):
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            results.append((text, score))
        
        return results


class InteractiveGenerator:
    """
    Interactive text generator with conversation context.
    """
    
    def __init__(self, generator: TextGenerator):
        """
        Initialize interactive generator.
        
        Args:
            generator: TextGenerator instance
        """
        self.generator = generator
        self.context = []
        
    def generate_response(
        self,
        prompt: str,
        max_length: int = 100,
        **kwargs
    ) -> str:
        """
        Generate response maintaining conversation context.
        
        Args:
            prompt: User input
            max_length: Maximum response length
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        # Build context
        self.context.append(prompt)
        full_prompt = " ".join(self.context[-10:])  # Keep last 10 turns
        
        # Generate
        responses = self.generator.generate(
            prompt=full_prompt,
            max_length=max_length,
            num_return_sequences=1,
            **kwargs
        )
        
        response = responses[0]
        self.context.append(response)
        
        return response
    
    def reset_context(self):
        """Clear conversation context."""
        self.context = []


def batch_generate(
    generator: TextGenerator,
    prompts: List[str],
    **generation_kwargs
) -> List[str]:
    """
    Generate text for multiple prompts.
    
    Args:
        generator: TextGenerator instance
        prompts: List of prompt strings
        **generation_kwargs: Generation parameters
        
    Returns:
        List of generated texts
    """
    results = []
    for prompt in prompts:
        generated = generator.generate(
            prompt=prompt,
            num_return_sequences=1,
            **generation_kwargs
        )
        results.append(generated[0])
    
    return results