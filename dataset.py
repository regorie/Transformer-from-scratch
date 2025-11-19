import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import json

PAD=0
EOS=1
BOS=2
UNK=3

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        tok2id = json.load(f)

    return tok2id

def text_to_ids(text, vocab, add_special_tokens=True):
    """Convert BPE text to sequence of IDs"""
    tokens = text.strip().split()
    
    if add_special_tokens:
        tokens = ['<BOS>'] + tokens + ['<EOS>']
    
    # Convert to IDs, use <unk> for unknown tokens
    ids = []
    for token in tokens:
        ids.append(vocab.get(token, vocab['<UNK>']))
    
    return ids


class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, vocab, max_length=512):
        self.src_sentences = []
        self.trg_sentences = []
        self.vocab = vocab
        self.max_length = max_length
        
        # Load and convert sentences
        src_token_counts = []
        trg_token_counts = []
        
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_ids = text_to_ids(src_line, vocab, add_special_tokens=False)
                trg_ids = text_to_ids(trg_line, vocab)
                
                # Skip very long sentences
                if len(src_ids) <= max_length and len(trg_ids) <= max_length:
                    self.src_sentences.append(src_ids)
                    self.trg_sentences.append(trg_ids)
                    src_token_counts.append(len(src_ids))
                    trg_token_counts.append(len(trg_ids))
        
        # Calculate average token counts
        self.avg_src_tokens = sum(src_token_counts) / len(src_token_counts) if src_token_counts else 0
        self.avg_trg_tokens = sum(trg_token_counts) / len(trg_token_counts) if trg_token_counts else 0
        self.avg_total_tokens = (self.avg_src_tokens + self.avg_trg_tokens) / 2
        
        print(f"Dataset statistics:")
        print(f"  Total sentences: {len(self.src_sentences)}")
        print(f"  Average source tokens: {self.avg_src_tokens:.2f}")
        print(f"  Average target tokens: {self.avg_trg_tokens:.2f}")
        print(f"  Average total tokens per sentence pair: {self.avg_total_tokens:.2f}")
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_sentences[idx], dtype=torch.long),
            'trg': torch.tensor(self.trg_sentences[idx], dtype=torch.long),
        }
    
def collate_fn(batch, pad_token_id=0):
    """Collate function to pad sequences in a batch"""
    src_sequences = [item['src'] for item in batch]
    trg_sequences = [item['trg'] for item in batch]
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_sequences, batch_first=True, padding_value=pad_token_id)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_sequences, batch_first=True, padding_value=pad_token_id)
    
    # Create attention masks (True for real tokens, False for padding)
    src_mask = (src_padded != pad_token_id)
    trg_mask = (trg_padded != pad_token_id)
    
    return {
        'src': src_padded, 
        'trg': trg_padded,
        'src_mask': src_mask, 
        'trg_mask': trg_mask
    }

def calculate_batch_size(avg_tokens, target_tokens_per_batch, gradient_accumulation_step):
    """
    Calculate appropriate batch size based on average tokens per sentence.
    
    Args:
        avg_tokens: Average number of tokens per sentence pair
        target_tokens_per_batch: Target number of tokens per batch (default: 8192)
    
    Returns:
        Recommended batch size
    """
    
    if avg_tokens <= 0:
        return 32  # Default batch size
    
    batch_size = max(1, int(target_tokens_per_batch / gradient_accumulation_step / avg_tokens))
    # Limit batch size to prevent memory issues and gradient explosion
    #batch_size = min(batch_size, 128)  # Cap at reasonable maximum
    effective_tokens = batch_size * gradient_accumulation_step * avg_tokens
    print(f"Recommended batch size: {batch_size} (avg tokens: {avg_tokens:.2f}, grad_accum: {gradient_accumulation_step}, effective tokens per update: {effective_tokens:.0f})")
    return batch_size

def create_dataloader(lang, vocab, token_per_batch=25000, batch_size=128, mode='train', gradient_accumulation_step=1, auto_batch_size=True):
    # lang example ['en', 'de']
    if vocab is None:
        vocab = load_vocab(f'./datasets/wmt14_{lang[0]}_{lang[1]}/vocab.json')

    if mode == 'train':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/train_{lang[0]}.BPE', f'./datasets/wmt14_{lang[0]}_{lang[1]}/train_{lang[1]}.BPE', vocab)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    elif mode == 'valid':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/valid_{lang[0]}.BPE', f'./datasets/wmt14_{lang[0]}_{lang[1]}/valid_{lang[1]}.BPE', vocab)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    elif mode == 'test':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/test_{lang[0]}.BPE', f'./datasets/wmt14_{lang[0]}_{lang[1]}/test_{lang[1]}.BPE', vocab)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return loader
