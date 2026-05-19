"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # The shape of Q is (bs, nh, Tq, dk)
    # The shape of K is (bs, nh, Tv, dk)
    # The shape of Q is (bs, nh, Tv, dv)
    # nh = number of heads

    d_k = Q.size(-1)
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Apply softmax along Tv dimension to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # Create mask where padding tokens are True
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    
    # Padding mask: True where token is padding
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    
    # Causal mask: prevent attending to future tokens
    # Lower triangular matrix: False for current and past, True for future
    tgt_sub_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool),
        diagonal=1
    )  # [tgt_len, tgt_len]
    
    # Combine both masks
    tgt_mask = tgt_pad_mask | tgt_sub_mask  # [batch, 1, tgt_len, tgt_len]
    
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        # Linear projections for Q, K, V
        # These are the W_Q,W_K,W_V matrices of all heads combined together
        # dm = nh x dv and dv = dk
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)    # [batch, seq_k, d_model]
        V = self.W_v(value)  # [batch, seq_k, d_model]
        
        # Split into multiple heads: [batch, seq, d_model] -> [batch, num_heads, seq, d_k]
        # Instead of 1 T x dm matrix and doing softmax on resulting 1 Tq, Tv matrix
        # We are dividing the matrix into h T x dm/h matrices and getting
        # h such Tq x Tv softmax probabs to each query and hence getting 
        # h different representations for each query token 
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Apply dropout to attention output
        # This dropout prevents Attention Concentration or overfocus on a single
        # token. It acts against the peakiness of softmax and regualarizes it.
        attn_output = self.dropout(attn_output)
        
        # Concatenate heads: [batch, num_heads, seq_q, d_k] -> [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection -> [batch, seq_q, d_model]
        output = self.W_o(attn_output)
        
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        # The dropout reduces the excessive dependency on the positional encodings
        # and some key words
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        # Apply first linear layer with ReLU activation
        x = F.relu(self.linear1(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply second linear layer
        x = self.linear2(x)
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Modern Pre-LN Layer
        # Note: We normalize x BEFORE passing it to attention, but add it back to unnormalized x
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, src_mask)
        x = x + self.dropout(attn_output)

        # Note: We normalize x BEFORE passing it to feed-forward, but add it back to unnormalized x
        norm_x2 = self.norm2(x)
        ff_output = self.feed_forward(norm_x2)
        x = x + self.dropout(ff_output)

        # Pre-LN offeres much stable training compared to post-LN originaly used
        # Sample code for post-LN:
        # attn_output = self.self_attn(x, x, x, src_mask)
        # x = self.norm1(x + self.dropout(attn_output)) # Norm happens AFTER the addition

        # ff_output = self.feed_forward(x)
        # x = self.norm2(x + self.dropout(ff_output)) # Norm happens AFTER the addition

        """ In pre-LN that is used here, the skip-connection is un-normalized unlike in post-LN
        Hence, it acts as a direct gradient highway. Therefore, there
        would be no problem of vanishing gradients."""
        
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        norm_x = self.norm1(x)
        self_attn_output = self.self_attn(norm_x, norm_x, norm_x, tgt_mask)
        x = x + self.dropout(self_attn_output)
                
        # Cross-attention with encoder memory
        # Note that memory would have already been normalized
        norm_x2 = self.norm2(x)
        cross_attn_output = self.cross_attn(norm_x2, memory, memory, src_mask)
        x = x + self.dropout(cross_attn_output)
        
        # Feed-forward with residual connection and layer norm
        norm_x3 = self.norm3(x)
        ff_output = self.feed_forward(norm_x3)
        x = x + self.dropout(ff_output)
        
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Create N copies of the encoder layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        # Pass through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)
        # Final layer normalization
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        # Create N copies of the decoder layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # Final layer normalization
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int = 18669,
        tgt_vocab_size: int = 9797,
        d_model:   int   = 256,
        N:         int   = 4,
        num_heads: int   = 8,
        d_ff:      int   = 1024,
        dropout:   float = 0.1,
        checkpoint_path: str = "g_best_model.pt",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Encoder and decoder stacks
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters with Xavier/Glorot initialization
        self._init_parameters()
        
        # Store config for checkpointing
        self.model_config = {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'd_model': d_model,
            'N': N,
            'num_heads': num_heads,
            'd_ff': d_ff,
            'dropout': dropout
        }
        
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                # Download from Google Drive
                print("downloading the model from gdrive...")
                gdown.download(id = "14RZsnA4XRMe16eVeK_6T9yeKnuxlKkA5", output=checkpoint_path, quiet=False)
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        

        # Load German spaCy tokenizer
        import spacy
        import subprocess
        import sys
        try:
            self._spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            # Use sys.executable to safely invoke the correct Python binary
            print("downloading spacy tokenizer for german...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "de_core_news_sm"], 
                check=True
            )
            self._spacy_de = spacy.load('de_core_news_sm')

    
    def _init_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Embed source tokens and scale by sqrt(d_model)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src_emb = self.pos_encoding(src_emb)
        
        # Pass through encoder
        memory = self.encoder(src_emb, src_mask)
        
        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Embed target tokens and scale by sqrt(d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Pass through decoder
        decoder_output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Encode source sequence
        memory = self.encode(src, src_mask)
        
        # Decode target sequence
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        
        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
         # Special token indices (must match dataset.py constants)
        PAD_IDX = 1
        SOS_IDX = 2
        EOS_IDX = 3
        UNK_IDX = 0
        MAX_LEN = 100
 
        # Lazily load vocab + tokenizer; cache on self so repeated
        #          calls to infer() don't reload the full dataset each time
        if not hasattr(self, '_src_vocab') or self._src_vocab is None:
            from dataset import Multi30kDataset
 
            # Build vocab from the training split (same vocab used during training)
            _ds = Multi30kDataset(split='train')
            self._src_vocab     = _ds.src_vocab        # token  --> idx  (German)
            self._tgt_idx2token = _ds.tgt_idx2token    # idx   --> token (English)

        # Tokenise raw German text
        tokens = [tok.text.lower() for tok in self._spacy_de.tokenizer(src_sentence)]
 
        # Map tokens --> indices (unknown tokens → UNK_IDX)
        src_indices = [self._src_vocab.get(tok, UNK_IDX) for tok in tokens]
 
        # Build source tensor [1, src_len] and its mask 
        device = next(self.parameters()).device
        src      = torch.tensor([src_indices], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
 
        #  Greedy autoregressive decoding 
        #          Logic mirrors greedy_decode() in train.py, inlined here so
        #          infer() is fully self-contained.
        self.eval()
        with torch.no_grad():
            # Encode the source sequence once
            memory = self.encode(src, src_mask)
 
            # Start with <sos>
            ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
 
            for _ in range(MAX_LEN - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=PAD_IDX).to(device)
 
                # logits: [1, current_tgt_len, tgt_vocab_size]
                logits = self.decode(memory, src_mask, ys, tgt_mask)
 
                # Greedy: pick the highest-probability token at the last position
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                ys = torch.cat([ys, next_token], dim=1)
 
                # Stop when <eos> is produced
                if next_token.item() == EOS_IDX:
                    break
 
        # Convert predicted indices -> English tokens
        #          Skip the leading <sos>; stop before <eos>.
        predicted_ids = ys[0].cpu().tolist()[1:]   # drop <sos>
        words = []
        for idx in predicted_ids:
            if idx == EOS_IDX:
                break
            words.append(self._tgt_idx2token.get(idx, '<unk>'))

        translated_string = " ".join(words)
        
        # Replace the problematic ' .' with '.'
        translated_string = translated_string.replace(" .", ".")
        
        return translated_string
