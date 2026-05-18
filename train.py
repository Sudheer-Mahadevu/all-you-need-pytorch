"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        with torch.no_grad():
            # Initialize with uniform smoothing
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # -2 for pad and true label
            
            # Set confidence for true labels
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Zero out padding positions
            true_dist[:, self.pad_idx] = 0
            
            # Mask out padding targets
            mask = (target == self.pad_idx)
            if mask.any():
                true_dist[mask] = 0
        
        # Compute KL divergence loss
        loss = -(true_dist * log_probs).sum(dim=-1)
        
        # Mask out padding positions in loss
        mask = (target != self.pad_idx)
        loss = loss * mask.float()
        
        # Return mean loss over non-padding tokens
        return loss.sum() / mask.sum()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    is_bkg: bool = False,
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    # Progress bar
    pbar = tqdm(data_iter, desc=f"{'Train' if is_train else 'Eval'} Epoch {epoch_num}", disable=is_bkg)
    
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Prepare input and target
        # tgt_input: all tokens except the last one
        # tgt_output: all tokens except the first one (shifted by 1)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        src_mask = make_src_mask(src, pad_idx=1).to(device)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=1).to(device)
        
        # Forward pass
        # logits shape : (batch, tgt_len, tgt_vocab_size)
        if is_train:
            logits = model(src, tgt_input, src_mask, tgt_mask)
        else:
            with torch.no_grad():
                logits = model(src, tgt_input, src_mask, tgt_mask)
        
        # Reshape for loss computation
        logits_flat = logits.contiguous().view(-1, logits.size(-1))
        tgt_flat = tgt_output.contiguous().view(-1)
        
        # Compute loss
        loss = loss_fn(logits_flat, tgt_flat)
        
        if is_train:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
        
        # Accumulate stats
        num_tokens = (tgt_output != 1).sum().item()  # Count non-padding tokens
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    
    # Encode the source sequence once
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        
        # Start with the start symbol
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            # Create target mask for current sequence
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            
            # Decode
            # shape of logits (1, tgt_len, tgt_vocab_size)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            
            # Get the last token's prediction (greedy)
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            
            # Append to output sequence
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if we predict end symbol
            if next_token.item() == end_symbol:
                break
    
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
    is_bkg : bool = False,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    from sacrebleu import corpus_bleu
    
    model.eval()
    
    references = []
    hypotheses = []
    
    for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU", disable=is_bkg):
        # Process each example in the batch individually
        for i in range(src.size(0)):
            src_seq = src[i:i+1].to(device)  # [1, src_len]
            tgt_seq = tgt[i]  # [tgt_len]
            
            # Create source mask
            src_mask = make_src_mask(src_seq, pad_idx=1).to(device)
            
            # Decode
            output = greedy_decode(
                model, src_seq, src_mask,
                max_len=max_len,
                start_symbol=2,  # <sos>
                end_symbol=3,    # <eos>
                device=device
            )
            
            # Convert indices to tokens
            # For hypothesis: skip <sos>, stop at <eos> or padding
            hyp_indices = output[0].cpu().tolist()[1:]  # Skip <sos>
            hyp_tokens = []
            for idx in hyp_indices:
                if idx == 3:  # <eos>
                    break
                if idx == 1:  # <pad>
                    break
                # Access vocabulary - try different methods
                if hasattr(tgt_vocab, 'itos'):
                    token = tgt_vocab.itos[idx]
                elif hasattr(tgt_vocab, 'lookup_token'):
                    token = tgt_vocab.lookup_token(idx)
                else:
                    # Assume it's a dict-like object
                    token = tgt_vocab.get(idx, '<unk>')
                hyp_tokens.append(token)
            
            # For reference: skip <sos>, stop at <eos> or padding
            ref_indices = tgt_seq.cpu().tolist()[1:]  # Skip <sos>
            ref_tokens = []
            for idx in ref_indices:
                if idx == 3:  # <eos>
                    break
                if idx == 1:  # <pad>
                    break
                if hasattr(tgt_vocab, 'itos'):
                    token = tgt_vocab.itos[idx]
                elif hasattr(tgt_vocab, 'lookup_token'):
                    token = tgt_vocab.lookup_token(idx)
                else:
                    token = tgt_vocab.get(idx, '<unk>')
                ref_tokens.append(token)
            
            # Join tokens into sentences
            hyp_sentence = ' '.join(hyp_tokens)
            ref_sentence = ' '.join(ref_tokens)
            
            hypotheses.append(hyp_sentence)
            references.append([ref_sentence])  # BLEU expects list of references
    
    # Compute corpus BLEU
    bleu = corpus_bleu(hypotheses, references)
    
    return bleu.score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'model_config': model.model_config
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Checkpoint loaded from {path}, epoch {epoch}")
    
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(config) -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    import wandb
    from dataset import Multi30kDataset, collate_fn
    from lr_scheduler import NoamScheduler
    from functools import partial
    
    # 2. Load datasets
    print("Loading datasets...")
    train_dataset = Multi30kDataset(split='train')
    val_dataset = Multi30kDataset(split='validation')
    test_dataset = Multi30kDataset(split='test')
    
    # Share vocabularies across splits
    val_dataset.src_vocab = train_dataset.src_vocab
    val_dataset.tgt_vocab = train_dataset.tgt_vocab
    val_dataset.src_idx2token = train_dataset.src_idx2token
    val_dataset.tgt_idx2token = train_dataset.tgt_idx2token
    
    test_dataset.src_vocab = train_dataset.src_vocab
    test_dataset.tgt_vocab = train_dataset.tgt_vocab
    test_dataset.src_idx2token = train_dataset.src_idx2token
    test_dataset.tgt_idx2token = train_dataset.tgt_idx2token
    
    # Reprocess with shared vocab
    val_dataset.process_data()
    test_dataset.process_data()
    
    # 3. Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=partial(collate_fn, pad_idx=1)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=1)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=partial(collate_fn, pad_idx=1)
    )
    
    # 4. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. Create optimizer (Adam with specific betas and epsilon)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # 6. Create learning rate scheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps']
    )
    
    # 7. Create loss function
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=1,
        smoothing=config['smoothing']
    )
    
    # 8. Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch + 1,
            is_train=True,
            device=device,
            is_bkg= config['is_background']
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss})
        
        # Validate
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch_num=epoch + 1,
            is_train=False,
            device=device
        )

        print(f"Val Loss: {val_loss:.4f}")
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']

        wandb.log({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'learning_rate': current_lr,
        })

        print(f"\n  ┌{'─'*42}┐")
        print(f"  │  {'Epoch Summary':^40}  │")
        print(f"  ├{'─'*42}┤")
        print(f"  │  {'Epoch':<20} {epoch + 1:>4} / {config['num_epochs']:<4}        │")
        print(f"  │  {'Train Loss':<20} {train_loss:>10.4f}            │")
        print(f"  │  {'Val   Loss':<20} {val_loss:>10.4f}            │")
        print(f"  │  {'Learning Rate':<20} {current_lr:>10.2e}            │")
        print(f"  └{'─'*42}┘")
        
        # # Save checkpoint
        # checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
        # save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, "best_model.pt")
            print(f"New best model saved with val_loss: {val_loss:.4f}")
    
    # 9. Final evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    bleu_score = evaluate_bleu(
        model,
        test_loader,
        train_dataset.tgt_idx2token,
        device=device,
        max_len=100,
        is_bkg= config['is_background'],
    )
    
    print(f"Test BLEU Score: {bleu_score:.2f}")
    wandb.log({'test_bleu': bleu_score})
    
    wandb.finish()


if __name__ == "__main__":
    # Hyperparameters - adjusted for Multi30k dataset
    config = {
        'd_model': 256,      # Reduced from 512 for smaller dataset
        'N': 4,              # Reduced from 6 layers
        'num_heads': 8,
        'd_ff': 1024,        # Reduced from 2048
        'dropout': 0.1,
        'warmup_steps': 4000,
        'num_epochs': 15,
        'batch_size': 8,
        'smoothing': 0.1,
        'learning_rate': 1.0,  # Base LR for Noam scheduler\
        'run_name' : "trial_run",
        'is_background': True
    }

    import wandb
    from api_keys import WANDB_API_KEY, WANDB_ENTITY
    
    # 1. Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    wandb.init(entity=WANDB_ENTITY, project="da6401-a3", config=config,
               tags=["training"], name=config['run_name'])
    run_training_experiment(config)
