import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from collections import Counter

class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        self.dataset = load_dataset('bentrevett/multi30k', split=split)
        
        # Load spacy tokenizers for de and en
        import sys
        from unittest.mock import MagicMock
        
        # 1. Force the system to think 'google.colab' is already imported 
        #    and points to a dummy object. This stops spacy from 
        #    triggering the buggy import.
        sys.modules["google.colab"] = MagicMock()
        import spacy
        self.spacy_de = spacy.blank('de')
        self.spacy_en = spacy.blank('en')
        
        # Special tokens
        self.PAD_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        self.UNK_IDX = 0
        
        # Build vocabularies
        self.src_vocab = None
        self.tgt_vocab = None
        self.src_idx2token = None
        self.tgt_idx2token = None
        
        self.build_vocab()
        self.process_data()

    def tokenize_de(self, text):
        """Tokenize German text using spacy"""
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
    
    def tokenize_en(self, text):
        """Tokenize English text using spacy"""
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # Create the vocabulary dictionaries or torchtext Vocab equivalent
        src_counter = Counter()
        tgt_counter = Counter()
        
        # Count tokens in the dataset
        # German to English Translation is being learnt
        # Hence, src is german and tgt is english
        for example in self.dataset:
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])
            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)
        
        # Build vocab: special tokens first, then by frequency
        special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        # Source vocabulary (German)
        self.src_vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for idx, (token, _) in enumerate(src_counter.most_common(), start=len(special_tokens)):
            self.src_vocab[token] = idx
        
        # Target vocabulary (English)
        self.tgt_vocab = {token: idx for idx, token in enumerate(special_tokens)}
        for idx, (token, _) in enumerate(tgt_counter.most_common(), start=len(special_tokens)):
            self.tgt_vocab[token] = idx
        
        # Create reverse mappings for decoding
        self.src_idx2token = {idx: token for token, idx in self.src_vocab.items()}
        self.tgt_idx2token = {idx: token for token, idx in self.tgt_vocab.items()}
        
        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # Tokenize and convert words to indices
        self.processed_data = []
        
        for example in self.dataset:
            src_tokens = self.tokenize_de(example['de'])
            tgt_tokens = self.tokenize_en(example['en'])
            
            # Convert to indices, using UNK_IDX for unknown tokens
            src_indices = [self.src_vocab.get(token, self.UNK_IDX) for token in src_tokens]
            tgt_indices = [self.tgt_vocab.get(token, self.UNK_IDX) for token in tgt_tokens]
            
            # Add SOS and EOS tokens to target
            tgt_indices = [self.SOS_IDX] + tgt_indices + [self.EOS_IDX]
            
            self.processed_data.append({
                'src': src_indices,
                'tgt': tgt_indices
            })
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch, pad_idx=1):
    """
    Custom collate function to pad sequences in a batch.
    
    Args:
        batch: List of dictionaries with 'src' and 'tgt' keys
        pad_idx: Padding token index
    
    Returns:
        src_batch: Padded source sequences [batch_size, max_src_len]
        tgt_batch: Padded target sequences [batch_size, max_tgt_len]
    """
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    
    # Find max lengths
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)
    
    # Pad sequences
    src_padded = []
    tgt_padded = []
    
    for src, tgt in zip(src_batch, tgt_batch):
        src_padded.append(src + [pad_idx] * (max_src_len - len(src)))
        tgt_padded.append(tgt + [pad_idx] * (max_tgt_len - len(tgt)))
    
    return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)
