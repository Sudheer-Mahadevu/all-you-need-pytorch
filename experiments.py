"""
experiments.py  —  DA6401 Assignment 3, Section 2: W&B Report Experiments
==========================================================================

Five independent experiments covering:
  Q2.1  Noam Scheduler vs Fixed Learning Rate
  Q2.2  Ablation: Scaling Factor  1/√dₖ
  Q2.3  Attention Rollout & Head Specialization
  Q2.4  Sinusoidal vs Learned Positional Encoding
  Q2.5  Label Smoothing Ablation

Design rules
  • All plots are native W&B interactive charts (line charts, heatmaps, tables).
    No static matplotlib figures are logged.
  • Model-code changes required only for a specific experiment are documented in
    comments directly below the relevant experiment function — NOT in model.py.
  • The only global change to model.py is storing `self.attn_weights` in
    MultiHeadAttention (needed by Q2.3 and marked there).

Usage
-----
    python experiments.py --exp 2.1
    python experiments.py --exp 2.2
    python experiments.py --exp 2.3 --checkpoint best_model.pt
    python experiments.py --exp 2.4
    python experiments.py --exp 2.5
    python experiments.py --exp all           # runs 2.1 → 2.5 sequentially
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

# ── local imports ──────────────────────────────────────────────────────
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from model import (
    Transformer,
    MultiHeadAttention,
    make_src_mask,
    make_tgt_mask,
)
from train import (
    LabelSmoothingLoss,
    run_epoch,
    evaluate_bleu,
    save_checkpoint,
    load_checkpoint,
)

# ══════════════════════════════════════════════════════════════════════
#  GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════

# Base hyperconfig — shared across all experiments for fair comparison
BASE_CONFIG: dict = {
    "d_model":       256,
    "N":             4,
    "num_heads":     8,
    "d_ff":          1024,
    "dropout":       0.1,
    "warmup_steps":  4000,
    "num_epochs":    30,
    "batch_size":    64,
    "smoothing":     0.1,
    "learning_rate": 1.0,   # base LR; Noam multiplies this
}

WANDB_PROJECT = "da6401-a3"

# ══════════════════════════════════════════════════════════════════════
#  SHARED DATASET / DATALOADER HELPERS
# ══════════════════════════════════════════════════════════════════════

def load_datasets() -> tuple:
    """Load train / val / test with shared vocabulary (train vocab is canonical)."""
    print("Loading Multi30k datasets …")
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    for ds in (val_ds, test_ds):
        ds.src_vocab     = train_ds.src_vocab
        ds.tgt_vocab     = train_ds.tgt_vocab
        ds.src_idx2token = train_ds.src_idx2token
        ds.tgt_idx2token = train_ds.tgt_idx2token
        ds.process_data()

    return train_ds, val_ds, test_ds


def make_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    pad = partial(collate_fn, pad_idx=1)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=pad)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=pad)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=pad)
    return train_dl, val_dl, test_dl


# ══════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ══════════════════════════════════════════════════════════════════════

def build_standard_model(train_ds, config: dict, device) -> Transformer:
    """Construct a fresh Transformer (no checkpoint load, no spaCy for infer)."""
    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,   #  don't auto-download/load weights
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ══════════════════════════════════════════════════════════════════════
#  GENERIC TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_loop(
    run_name:       str,
    config:         dict,
    train_dl,
    val_dl,
    test_dl,
    train_ds,
    device,
    model:          nn.Module,
    fixed_lr:       Optional[float] = None,   # Q2.1: constant LR, no Noam
    log_grad_norms: bool = False,              # Q2.2: W_q/W_k grad norms ≤ 1000 steps
    log_confidence: bool = False,              # Q2.5: per-step prediction confidence
    wandb_group:    Optional[str] = None,
    extra_tags:     Optional[list] = None,
    save_best_as:   Optional[str] = None,
) -> tuple:
    """
    Generic training loop that routes all scalar metrics through wandb.log so
    W&B auto-renders interactive line charts.

    Parameters
    ----------
    fixed_lr : float | None
        When given, uses Adam at this constant rate with no scheduler (Q2.1).
    log_grad_norms : bool
        When True, logs W_q / W_k gradient norms for the first 1 000 steps
        across all encoder layers (Q2.2).
    log_confidence : bool
        When True, logs the mean softmax probability of the correct token per
        training step (Q2.5).

    Returns
    -------
    model       : trained model
    bleu_score  : corpus BLEU on the test set (float)
    """
    tags = ["section2"] + (extra_tags or [])
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
        group=wandb_group,
        tags=tags,
        reinit=True,
    )

    # ── Optimizer ──────────────────────────────────────────────────────
    lr = fixed_lr if fixed_lr is not None else config["learning_rate"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9
    )

    # ── Scheduler ──────────────────────────────────────────────────────
    if fixed_lr is None:
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    else:
        scheduler = None   # constant LR experiment

    # ── Loss ────────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_ds.tgt_vocab),
        pad_idx=1,
        smoothing=config.get("smoothing", 0.1),
    )

    best_val_loss = float("inf")
    global_step   = 0

    # ── Epoch loop ──────────────────────────────────────────────────────
    for epoch in range(config["num_epochs"]):

        # ── TRAIN ───────────────────────────────────────────────────────
        model.train()
        epoch_loss_sum = 0.0
        epoch_tokens   = 0

        for src, tgt in train_dl:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=1).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=1).to(device)

            logits      = model(src, tgt_in, src_mask, tgt_mask)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))
            tgt_flat    = tgt_out.contiguous().view(-1)

            loss = loss_fn(logits_flat, tgt_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ntok            = (tgt_out != 1).sum().item()
            epoch_loss_sum += loss.item() * ntok
            epoch_tokens   += ntok
            global_step    += 1

            # ── per-step W&B log ────────────────────────────────────────
            step_log = {
                "step":      global_step,
                "step_loss": loss.item(),
                "lr":        optimizer.param_groups[0]["lr"],
            }

            # Q2.2 — gradient norms for W_q and W_k (first 1 000 steps only)
            if log_grad_norms and global_step <= 1000:
                for li, layer in enumerate(model.encoder.layers):
                    wq_g = layer.self_attn.W_q.weight.grad
                    wk_g = layer.self_attn.W_k.weight.grad
                    if wq_g is not None:
                        step_log[f"grad_norm/enc_L{li}_Wq"] = wq_g.norm().item()
                    if wk_g is not None:
                        step_log[f"grad_norm/enc_L{li}_Wk"] = wk_g.norm().item()

            # Q2.5 — prediction confidence (mean softmax prob of correct token)
            if log_confidence:
                with torch.no_grad():
                    probs     = F.softmax(logits_flat.detach(), dim=-1)
                    pad_mask  = tgt_flat != 1
                    if pad_mask.any():
                        correct_p = probs[pad_mask].gather(
                            1, tgt_flat[pad_mask].unsqueeze(1)
                        ).squeeze(1)
                        step_log["prediction_confidence"] = correct_p.mean().item()

            wandb.log(step_log)

        train_loss = epoch_loss_sum / epoch_tokens if epoch_tokens > 0 else 0.0

        # ── EVAL ────────────────────────────────────────────────────────
        val_loss = run_epoch(
            val_dl, model, loss_fn,
            optimizer=None, scheduler=None,
            epoch_num=epoch + 1, is_train=False,
            device=device, is_bkg=True,
        )

        # Epoch-level log — W&B auto-creates overlay line charts when multiple
        # runs share the same metric keys inside the same project group.
        wandb.log({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

        print(
            f"[{run_name}] epoch {epoch+1:>2}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ── checkpoint ──────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_best_as or f"best_{run_name}.pt"
            save_checkpoint(
                model,
                optimizer,
                scheduler if scheduler else optimizer,  # dummy fallback
                epoch + 1,
                ckpt_path,
            )
            print(f"  → checkpoint saved  (val_loss={val_loss:.4f})")

    # ── Test BLEU ───────────────────────────────────────────────────────
    bleu = evaluate_bleu(
        model, test_dl, train_ds.tgt_idx2token,
        device=device, is_bkg=True,
    )
    wandb.log({"test_bleu": bleu})
    print(f"[{run_name}]  Test BLEU = {bleu:.2f}")

    wandb.finish()
    return model, bleu


# ══════════════════════════════════════════════════════════════════════
#  Q2.2 — Variant: Unscaled Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════
# Changes required vs model.py:
#
#   In scaled_dot_product_attention(), REPLACE:
#       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
#   WITH:
#       scores = torch.matmul(Q, K.transpose(-2, -1))   # no sqrt(d_k) scaling
#
# Rather than touching model.py, we subclass MultiHeadAttention and
# override forward() to skip the scaling — see UnscaledMHA below.
# ─────────────────────────────────────────────────────────────────────

class UnscaledMHA(MultiHeadAttention):
    """
    Drop-in replacement for MultiHeadAttention that skips the 1/√dₖ factor.
    Used exclusively in Q2.2 to isolate the effect of the scaling term.
    """

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key  ).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # ── NO SCALING: raw dot-product without dividing by √dₖ ──────────
        scores = torch.matmul(Q, K.transpose(-2, -1))   # shape [..., Tq, Tv]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        self.attn_weights = F.softmax(scores, dim=-1)   # store for inspection

        attn_output = self.dropout(torch.matmul(self.attn_weights, V))
        attn_output = (
            attn_output.transpose(1, 2).contiguous()
            .view(batch_size, -1, self.d_model)
        )
        return self.W_o(attn_output)


def _swap_mha_to_unscaled(model: Transformer) -> Transformer:
    """Replace every MultiHeadAttention in a model with UnscaledMHA in-place."""
    cfg = model.model_config

    def new_mha():
        return UnscaledMHA(cfg["d_model"], cfg["num_heads"], cfg["dropout"])

    for layer in model.encoder.layers:
        layer.self_attn = new_mha()
    for layer in model.decoder.layers:
        layer.self_attn  = new_mha()
        layer.cross_attn = new_mha()
    return model


# ══════════════════════════════════════════════════════════════════════
#  Q2.4 — Variant: Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════
# Changes required vs model.py:
#
#   In Transformer.__init__(), REPLACE:
#       self.pos_encoding = PositionalEncoding(d_model, dropout)
#   WITH:
#       self.pos_encoding = LearnedPositionalEncoding(d_model, dropout, max_len=256)
#
#   The LearnedPositionalEncoding class is defined just below.
#   The Transformer subclass TransformerLearnedPE below wraps this swap.
# ─────────────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding via nn.Embedding (Q2.4 variant).

    Replaces the deterministic sinusoidal formula with a trainable
    lookup table of shape [max_len, d_model].

    Note: Unlike sinusoidal encoding, this cannot extrapolate to sequence
    lengths > max_len seen during training (see Q2.4 analysis).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.max_len   = max_len
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]  (x + learned positional vectors)
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_len={self.max_len} for learned PE."
            )
        positions = torch.arange(seq_len, device=x.device)  # [seq_len]
        return self.dropout(x + self.embedding(positions))


class TransformerLearnedPE(Transformer):
    """
    Transformer whose positional encoding is a learned nn.Embedding.
    All other components are identical to the base Transformer.
    """

    def __init__(self, max_pe_len: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        d_model  = kwargs.get("d_model", BASE_CONFIG["d_model"])
        dropout  = kwargs.get("dropout", BASE_CONFIG["dropout"])
        # Override the sinusoidal PE with learned embeddings
        self.pos_encoding = LearnedPositionalEncoding(d_model, dropout, max_pe_len)


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2.1 — Noam Scheduler vs Fixed Learning Rate
# ══════════════════════════════════════════════════════════════════════

def run_q21_noam_vs_fixed_lr(train_ds, val_ds, test_ds, device: str) -> None:
    """
    Train the same architecture twice:
      Run A — Noam schedule  (linear warm-up → inverse sqrt decay)
      Run B — Constant LR = 1e-4  (no warm-up, no decay)

    Both runs are placed in the W&B group "q21_scheduler_comparison".
    In the W&B UI, open the group and overlay 'train_loss' and 'val_loss'
    across runs to see the divergence / instability of the fixed-LR model.

    W&B plots produced (native line charts, auto-created on logging):
      • step_loss  vs step
      • train_loss / val_loss  vs epoch
      • lr  vs step  (flat for fixed, ramp+decay for Noam)
    """
    cfg = dict(BASE_CONFIG, experiment="q2.1")
    train_dl, val_dl, test_dl = make_dataloaders(train_ds, val_ds, test_ds, cfg["batch_size"])

    # ── Run A: Noam ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.1 — Run A: Noam Scheduler")
    print("=" * 60)
    model_a = build_standard_model(train_ds, cfg, device)
    train_loop(
        run_name="q21_noam",
        config=dict(cfg, scheduler="noam"),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_a,
        fixed_lr=None,                       # use Noam
        wandb_group="q21_scheduler_comparison",
        extra_tags=["q2.1", "noam"],
        save_best_as="best_q21_noam.pt",
    )

    # ── Run B: Fixed LR ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.1 — Run B: Fixed LR = 1e-4")
    print("=" * 60)
    model_b = build_standard_model(train_ds, cfg, device)
    train_loop(
        run_name="q21_fixed_lr",
        config=dict(cfg, scheduler="fixed", fixed_lr=1e-4),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_b,
        fixed_lr=1e-4,                       # constant LR
        wandb_group="q21_scheduler_comparison",
        extra_tags=["q2.1", "fixed_lr"],
        save_best_as="best_q21_fixed.pt",
    )

    print("\n✓  Q2.1 done.  Overlay the two runs in W&B → group 'q21_scheduler_comparison'.")


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2.2 — Ablation: 1/√dₖ Scaling Factor
# ══════════════════════════════════════════════════════════════════════

def run_q22_scaling_ablation(train_ds, val_ds, test_ds, device: str) -> None:
    """
    Train two models:
      Run A — Standard Transformer with 1/√dₖ scaling (baseline)
      Run B — Same architecture but WITHOUT the scaling term

    Gradient norms of W_q and W_k are logged every step for the first
    1 000 training steps for both runs.  In W&B, compare:
      grad_norm/enc_L*_Wq  and  grad_norm/enc_L*_Wk

    W&B plots produced:
      • grad_norm/enc_L{0..N-1}_Wq  vs step  (native line chart)
      • grad_norm/enc_L{0..N-1}_Wk  vs step  (native line chart)
      • step_loss  vs step
      • train_loss / val_loss  vs epoch

    NOTE on model-code changes
    ──────────────────────────
    The unscaled variant is implemented here via UnscaledMHA (subclasses
    MultiHeadAttention and skips the /√dₖ in forward()).
    In model.py the equivalent change would be in
    scaled_dot_product_attention():
        CHANGE:   scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        TO:       scores = torch.matmul(Q, K.transpose(-2,-1))
    We avoid touching model.py so the autograder contract is intact.
    """
    cfg = dict(BASE_CONFIG, experiment="q2.2")
    train_dl, val_dl, test_dl = make_dataloaders(train_ds, val_ds, test_ds, cfg["batch_size"])

    # ── Run A: with scaling ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.2 — Run A: With 1/√dₖ scaling (baseline)")
    print("=" * 60)
    model_a = build_standard_model(train_ds, cfg, device)
    train_loop(
        run_name="q22_with_scaling",
        config=dict(cfg, use_scaling=True),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_a,
        log_grad_norms=True,
        wandb_group="q22_scaling_ablation",
        extra_tags=["q2.2", "scaled"],
        save_best_as="best_q22_scaled.pt",
    )

    # ── Run B: without scaling ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.2 — Run B: WITHOUT 1/√dₖ scaling")
    print("=" * 60)
    model_b = build_standard_model(train_ds, cfg, device)
    model_b = _swap_mha_to_unscaled(model_b).to(device)   # replace all MHA modules
    train_loop(
        run_name="q22_no_scaling",
        config=dict(cfg, use_scaling=False),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_b,
        log_grad_norms=True,
        wandb_group="q22_scaling_ablation",
        extra_tags=["q2.2", "unscaled"],
        save_best_as="best_q22_unscaled.pt",
    )

    print("\n✓  Q2.2 done.  Inspect grad_norm/* plots in W&B group 'q22_scaling_ablation'.")


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2.3 — Attention Rollout & Head Specialization
# ══════════════════════════════════════════════════════════════════════

def run_q23_attention_rollout(
    train_ds,
    device: str,
    checkpoint_path: str = "best_model.pt",
) -> None:
    """
    Load a trained checkpoint, pass a single German source sentence through
    the encoder, and log the per-head attention heatmaps from every encoder
    layer (emphasis on the last layer) as native W&B HeatMap charts.

    Additionally computes Attention Rollout: the product of row-normalised
    attention matrices (+ residual identity) across layers, giving a measure
    of total information flow from input tokens.

    W&B plots produced (native wandb.plots.HeatMap interactive charts):
      • attn/layer_{l}_head_{h}         — per-layer, per-head heatmap
      • attn/last_layer_head_{h}        — last layer, all heads
      • attn/rollout_mean_heads         — attention rollout (layer-folded)

    NOTE on model-code changes
    ──────────────────────────
    This experiment relies on the GLOBAL CHANGE already applied to model.py:
      MultiHeadAttention.__init__:  self.attn_weights = None
      MultiHeadAttention.forward:   self.attn_weights = <weights from sdpa>
    After any model.encode() or model.forward() call, per-layer attention
    weights are accessible as:
      model.encoder.layers[l].self_attn.attn_weights   # [B, H, T, T]
    No further changes to model.py are needed.
    """
    cfg = dict(BASE_CONFIG, experiment="q2.3")

    # ── Load model ────────────────────────────────────────────────────
    model = build_standard_model(train_ds, cfg, device)
    epoch = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")

    # ── Tokenise a fixed German source sentence ───────────────────────
    # Using a sentence from the Multi30k validation set — keep it short
    # enough so the heatmap is readable (≤ 20 tokens ideal).
    SAMPLE_DE = (
        "Ein Mann mit einem orangefarbenen Hut schaut in die Kamera ."
    )

    import sys
    from unittest.mock import MagicMock
        
    # 1. Force the system to think 'google.colab' is already imported 
    #    and points to a dummy object. This stops spacy from 
    #    triggering the buggy import.
    sys.modules["google.colab"] = MagicMock()
    import spacy
    spacy_de = spacy.blank('de')

    tokens = [tok.text.lower() for tok in spacy_de.tokenizer(SAMPLE_DE)]
    print(f"Source tokens: {tokens}")

    src_indices = [train_ds.src_vocab.get(t, 0) for t in tokens]
    src  = torch.tensor([src_indices], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, pad_idx=1).to(device)

    # ── Forward pass — encoder only ───────────────────────────────────
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    # ── Collect attention weights from all encoder layers ─────────────
    # Shape of each: [1, num_heads, seq_len, seq_len]
    num_layers = len(model.encoder.layers)
    num_heads  = cfg["num_heads"]
    seq_len    = len(tokens)

    all_attn = []   # list of [num_heads, seq_len, seq_len] (numpy)
    for l, layer in enumerate(model.encoder.layers):
        w = layer.self_attn.attn_weights   # [1, H, T, T]
        if w is None:
            raise RuntimeError(
                "attn_weights is None — check that the global change to "
                "model.py (storing self.attn_weights) was applied."
            )
        # Average over batch dim (here batch=1), keep on CPU
        w_np = w[0].cpu().float().numpy()  # [H, T, T]
        all_attn.append(w_np)

    # ── W&B init for this experiment ──────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name="q23_attention_rollout",
        config=dict(cfg, checkpoint=checkpoint_path, sample_sentence=SAMPLE_DE),
        tags=["section2", "q2.3"],
        reinit=True,
    )

    # ── Log per-head heatmaps (all layers) as W&B interactive charts ──
    #    wandb.plots.HeatMap renders a native interactive heatmap in the
    #    W&B UI — no static image is created.
    token_labels = tokens  # x- and y-axis labels

    for l in range(num_layers):
        for h in range(num_heads):
            matrix = all_attn[l][h].tolist()    # list-of-lists required by API
            wandb.log({
                f"attn/layer_{l}_head_{h}": wandb.plots.HeatMap(
                    x_labels=token_labels,
                    y_labels=token_labels,
                    matrix_values=matrix,
                    show_text=False,
                )
            })

    # ── Log last-layer heads separately for easy inspection ───────────
    last_attn = all_attn[-1]   # [H, T, T]
    for h in range(num_heads):
        wandb.log({
            f"attn/last_layer_head_{h}": wandb.plots.HeatMap(
                x_labels=token_labels,
                y_labels=token_labels,
                matrix_values=last_attn[h].tolist(),
                show_text=False,
            )
        })

    # ── Attention Rollout across layers ───────────────────────────────
    # Algorithm (Abnar & Zuidema, 2020):
    #   1. For each layer, average attention over heads.
    #   2. Add residual: A_hat = 0.5 * A + 0.5 * I   (models skip-connections)
    #   3. Re-normalise rows to sum to 1.
    #   4. Multiply matrices across layers: rollout = A_hat_L @ … @ A_hat_1
    import numpy as np

    rollout = np.eye(seq_len)   # start with identity
    for l in range(num_layers):
        avg_attn  = all_attn[l].mean(axis=0)           # [T, T], mean over heads
        aug_attn  = 0.5 * avg_attn + 0.5 * np.eye(seq_len)   # add residual
        row_sums  = aug_attn.sum(axis=-1, keepdims=True)
        aug_attn  = aug_attn / (row_sums + 1e-9)
        rollout   = aug_attn @ rollout

    wandb.log({
        "attn/rollout_mean_heads": wandb.plots.HeatMap(
            x_labels=token_labels,
            y_labels=token_labels,
            matrix_values=rollout.tolist(),
            show_text=False,
        )
    })

    # ── Log a W&B Table for per-head statistics ────────────────────────
    # Logs the "entropy" of each head's attention distribution — lower
    # entropy = more focused / specialised head.
    head_table = wandb.Table(columns=["layer", "head", "mean_entropy", "max_attn_weight"])
    for l in range(num_layers):
        for h in range(num_heads):
            attn_h  = all_attn[l][h]   # [T, T]
            # Entropy of each row's distribution, averaged over query positions
            ent = -np.sum(
                attn_h * np.log(attn_h + 1e-9), axis=-1
            ).mean()
            head_table.add_data(l, h, float(ent), float(attn_h.max()))
    wandb.log({"attn/head_statistics": head_table})

    wandb.finish()
    print("\n✓  Q2.3 done.  View 'attn/*' charts in W&B run 'q23_attention_rollout'.")


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2.4 — Sinusoidal vs Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════

def run_q24_positional_encoding(train_ds, val_ds, test_ds, device: str) -> None:
    """
    Train two models:
      Run A — Standard sinusoidal positional encoding (baseline)
      Run B — Learned positional encoding (nn.Embedding, max_len=256)

    W&B plots produced:
      • train_loss / val_loss  vs epoch  (overlaid across both runs)
      • test_bleu              (table at the end of each run)

    NOTE on model-code changes
    ──────────────────────────
    The learned PE variant is implemented via TransformerLearnedPE (defined
    above), which subclasses Transformer and overrides self.pos_encoding
    after super().__init__() with a LearnedPositionalEncoding module.

    The equivalent change in model.py would be in Transformer.__init__():
        CHANGE:
            self.pos_encoding = PositionalEncoding(d_model, dropout)
        TO:
            self.pos_encoding = LearnedPositionalEncoding(d_model, dropout, max_len=256)
    where LearnedPositionalEncoding is the class defined in experiments.py.
    """
    cfg = dict(BASE_CONFIG, experiment="q2.4")
    train_dl, val_dl, test_dl = make_dataloaders(train_ds, val_ds, test_ds, cfg["batch_size"])

    # ── Run A: Sinusoidal PE ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.4 — Run A: Sinusoidal Positional Encoding (baseline)")
    print("=" * 60)
    model_a = build_standard_model(train_ds, cfg, device)
    _, bleu_sin = train_loop(
        run_name="q24_sinusoidal_pe",
        config=dict(cfg, pe_type="sinusoidal"),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_a,
        wandb_group="q24_pe_comparison",
        extra_tags=["q2.4", "sinusoidal_pe"],
        save_best_as="best_q24_sinusoidal.pt",
    )

    # ── Run B: Learned PE ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.4 — Run B: Learned Positional Encoding")
    print("=" * 60)
    model_b = TransformerLearnedPE(
        max_pe_len=256,
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=cfg["d_model"],
        N=cfg["N"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        checkpoint_path=None,
    ).to(device)
    _, bleu_learned = train_loop(
        run_name="q24_learned_pe",
        config=dict(cfg, pe_type="learned"),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_b,
        wandb_group="q24_pe_comparison",
        extra_tags=["q2.4", "learned_pe"],
        save_best_as="best_q24_learned.pt",
    )

    # ── Summary table (W&B) ───────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name="q24_pe_summary",
        tags=["section2", "q2.4", "summary"],
        reinit=True,
    )
    summary = wandb.Table(columns=["PE type", "Test BLEU"])
    summary.add_data("Sinusoidal (fixed)", round(bleu_sin, 2))
    summary.add_data("Learned (nn.Embedding)", round(bleu_learned, 2))
    wandb.log({"q24/bleu_comparison": summary})
    wandb.finish()

    print(f"\n✓  Q2.4 done.  BLEU — sinusoidal: {bleu_sin:.2f}  learned: {bleu_learned:.2f}")
    print("   View 'q24_pe_comparison' group in W&B for loss curves and BLEU table.")


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2.5 — Label Smoothing Ablation
# ══════════════════════════════════════════════════════════════════════

def run_q25_label_smoothing(train_ds, val_ds, test_ds, device: str) -> None:
    """
    Train two models:
      Run A — Label smoothing  ε = 0.1  (paper default)
      Run B — No label smoothing  ε = 0.0  (standard cross-entropy)

    Per training step we log prediction_confidence: the mean softmax
    probability the model assigns to the correct gold token, computed
    only over non-padding positions.  Over-confident models will show a
    sharply increasing confidence curve; smoothed models stay lower.

    W&B plots produced:
      • prediction_confidence  vs step  (native line chart, per-step)
      • train_loss / val_loss  vs epoch
      • test_bleu
    """
    cfg = dict(BASE_CONFIG, experiment="q2.5")
    train_dl, val_dl, test_dl = make_dataloaders(train_ds, val_ds, test_ds, cfg["batch_size"])

    # ── Run A: smoothing = 0.1 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.5 — Run A: Label Smoothing ε = 0.1")
    print("=" * 60)
    model_a = build_standard_model(train_ds, cfg, device)
    train_loop(
        run_name="q25_smooth_0.1",
        config=dict(cfg, smoothing=0.1),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_a,
        log_confidence=True,
        wandb_group="q25_label_smoothing",
        extra_tags=["q2.5", "smooth_0.1"],
        save_best_as="best_q25_smooth.pt",
    )

    # ── Run B: no smoothing ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Q2.5 — Run B: No Label Smoothing (ε = 0.0)")
    print("=" * 60)
    model_b = build_standard_model(
        train_ds, dict(cfg, smoothing=0.0), device
    )
    train_loop(
        run_name="q25_smooth_0.0",
        config=dict(cfg, smoothing=0.0),
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl,
        train_ds=train_ds, device=device,
        model=model_b,
        log_confidence=True,
        wandb_group="q25_label_smoothing",
        extra_tags=["q2.5", "smooth_0.0"],
        save_best_as="best_q25_no_smooth.pt",
    )

    print("\n✓  Q2.5 done.  Compare 'prediction_confidence' curves in W&B group "
          "'q25_label_smoothing'.")


# ══════════════════════════════════════════════════════════════════════
#  MAIN — parse args and dispatch
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DA6401 A3 — Section 2 W&B Experiments"
    )
    parser.add_argument(
        "--exp",
        required=True,
        choices=["2.1", "2.2", "2.3", "2.4", "2.5", "all"],
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_model.pt",
        help="Path to a trained checkpoint (used by --exp 2.3).",
    )
    parser.add_argument(
        "--wandb_key",
        default=None,
        help="W&B API key (alternatively set WANDB_API_KEY env var).",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        help="W&B entity / username.",
    )
    args = parser.parse_args()

    # ── W&B login ─────────────────────────────────────────────────────
    api_key = args.wandb_key or os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    # If neither is provided, wandb will prompt interactively.

    # ── Device ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset (loaded once for efficiency) ──────────────────────────
    need_data = args.exp in {"2.1", "2.2", "2.4", "2.5", "all"}
    train_ds = val_ds = test_ds = None
    if need_data or args.exp == "2.3":
        train_ds, val_ds, test_ds = load_datasets()

    # ── Dispatch ──────────────────────────────────────────────────────
    exps_to_run = ["2.1", "2.2", "2.3", "2.4", "2.5"] if args.exp == "all" else [args.exp]

    for exp in exps_to_run:
        if exp == "2.1":
            run_q21_noam_vs_fixed_lr(train_ds, val_ds, test_ds, device)
        elif exp == "2.2":
            run_q22_scaling_ablation(train_ds, val_ds, test_ds, device)
        elif exp == "2.3":
            run_q23_attention_rollout(train_ds, device, checkpoint_path=args.checkpoint)
        elif exp == "2.4":
            run_q24_positional_encoding(train_ds, val_ds, test_ds, device)
        elif exp == "2.5":
            run_q25_label_smoothing(train_ds, val_ds, test_ds, device)

    print("\n🎉  All requested experiments finished.")


if __name__ == "__main__":
    main()
