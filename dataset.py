import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from convert_to_IDs import text_to_ids
from build_vocab import load_vocab, PAD, UNK, BOS, EOS

class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, vocab_src, vocab_trg, max_length=512):
        self.src_sentences = []
        self.trg_sentences = []
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.max_length = max_length
        
        # Load and convert sentences
        src_token_counts = []
        trg_token_counts = []
        
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_ids = text_to_ids(src_line, vocab_src, add_special_tokens=False)
                trg_ids = text_to_ids(trg_line, vocab_trg)
                
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
    # TODO consider gradient accumulation method
    if avg_tokens <= 0:
        return 32  # Default batch size
    
    batch_size = max(1, int(target_tokens_per_batch / gradient_accumulation_step / avg_tokens))
    effective_tokens = batch_size * gradient_accumulation_step * avg_tokens
    print(f"Recommended batch size: {batch_size} (avg tokens: {avg_tokens:.2f}, grad_accum: {gradient_accumulation_step}, effective tokens per update: {effective_tokens:.0f})")
    return batch_size

def create_dataloader(lang, vocab_src=None, vocab_trg=None, token_per_batch=25000, batch_size=128, mode='train', gradient_accumulation_step=5, auto_batch_size=True):
    # lang example ['en', 'de']
    if vocab_src is None:
        vocab_src, _ = load_vocab(f'./datasets/wmt14_{lang[0]}_{lang[1]}/vocab_{lang[0]}.pkl')
    if vocab_trg is None:
        vocab_trg, _ = load_vocab(f'./datasets/wmt14_{lang[0]}_{lang[1]}/vocab_{lang[1]}.pkl')

    if mode == 'train':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/train.{lang[0]}', f'./datasets/wmt14_{lang[0]}_{lang[1]}/train.{lang[1]}', vocab_src, vocab_trg)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    elif mode == 'valid':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/valid.{lang[0]}', f'./datasets/wmt14_{lang[0]}_{lang[1]}/valid.{lang[1]}', vocab_src, vocab_trg)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    elif mode == 'test':
        dataset = TranslationDataset(f'./datasets/wmt14_{lang[0]}_{lang[1]}/test.{lang[0]}', f'./datasets/wmt14_{lang[0]}_{lang[1]}/test.{lang[1]}', vocab_src, vocab_trg)
        if auto_batch_size:
            batch_size = calculate_batch_size(dataset.avg_total_tokens, token_per_batch, gradient_accumulation_step)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return loader
