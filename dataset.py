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

def text_to_ids(text, vocab):
    """Convert BPE text to sequence of IDs"""
    tokens = text.strip().split()
    
    # Convert to IDs, use <unk> for unknown tokens
    ids = []
    for token in tokens:
        ids.append(vocab.get(token, vocab['<UNK>']))
    
    return ids

def add_special_tokens(tokens, start_tok, end_tok):
    # tokens list of tensors?
    for line in tokens:
        if start_tok: line = torch.tensor([start_tok]) + line
        if end_tok: line = line + torch.tensor([end_tok])
    return tokens


class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, vocab, max_length=100):
        self.src_sentences = []
        #self.trg_sentences = []
        self.trg_input_sentences = []
        self.trg_output_sentences = []
        self.vocab = vocab
        self.max_length = max_length
        
        # Load and convert sentences
        src_token_counts = []
        trg_token_counts = []
        skipped_count = 0
        
        with open(src_file, 'r') as f_src, open(trg_file, 'r') as f_trg:
            for src_line, trg_line in zip(f_src, f_trg):
                src_ids = text_to_ids(src_line, vocab)
                trg_ids = text_to_ids(trg_line, vocab)
                
                # STRICT length filtering to prevent memory issues
                if len(src_ids) <= max_length and len(trg_ids) <= max_length \
                    and len(src_ids) > 4 and len(trg_ids) > 4:
                    self.src_sentences.append(src_ids)
                    #self.trg_sentences.append(trg_ids)
                    self.trg_input_sentences.append([BOS]+trg_ids)
                    self.trg_output_sentences.append(trg_ids+[EOS])
                    src_token_counts.append(len(src_ids))
                    trg_token_counts.append(len(trg_ids))
                else:
                    skipped_count += 1
        
        # Calculate average token counts
        self.avg_src_tokens = sum(src_token_counts) / len(src_token_counts) if src_token_counts else 0
        self.avg_trg_tokens = sum(trg_token_counts) / len(trg_token_counts) if trg_token_counts else 0
        self.avg_total_tokens = (self.avg_src_tokens + self.avg_trg_tokens) / 2
        
        print(f"Dataset statistics:")
        print(f"  Total sentences: {len(self.src_sentences)}")
        print(f"  Skipped sentences (too long): {skipped_count}")
        print(f"  Max length limit: {max_length}")
        print(f"  Average source tokens: {self.avg_src_tokens:.2f}")
        print(f"  Average target tokens: {self.avg_trg_tokens:.2f}")
        print(f"  Average total tokens per sentence pair: {self.avg_total_tokens:.2f}")
        if src_token_counts:
            print(f"  Max source length: {max(src_token_counts)}")
            print(f"  Max target length: {max(trg_token_counts)}")
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_sentences[idx], dtype=torch.long),
            'trg_input': torch.tensor(self.trg_input_sentences[idx], dtype=torch.long),
            'trg_output': torch.tensor(self.trg_output_sentences[idx], dtype=torch.long),
        }
    
def collate_fn(batch, pad_token_id=0):
    """Collate function to pad sequences in a batch"""
    src_sequences = [item['src'] for item in batch]
    trg_input_sequences = [item['trg_input'] for item in batch]
    trg_output_sequences = [item['trg_output'] for item in batch]

    #trg_input = add_special_tokens(trg_sequences, BOS, None)
    #trg_output = add_special_tokens(trg_sequences, None, EOS)
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_sequences, batch_first=True, padding_value=pad_token_id)
    trg_input_padded = torch.nn.utils.rnn.pad_sequence(trg_input_sequences, batch_first=True, padding_value=pad_token_id)
    trg_output_padded = torch.nn.utils.rnn.pad_sequence(trg_output_sequences, batch_first=True, padding_value=pad_token_id)
    
    # Create attention masks (True for real tokens, False for padding)
    src_mask = (src_padded != pad_token_id)
    trg_input_mask = (trg_input_padded != pad_token_id) # False for padding
    #trg_output_mask = (trg_output_padded != pad_token_id)
    
    return {
        'src': src_padded, 
        'trg_input': trg_input_padded,
        'trg_output': trg_output_padded,
        'src_mask': src_mask, 
        'trg_input_mask': trg_input_mask,
        #'trg_output_mask': trg_output_mask
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
    
    # Calculate batch size to achieve target tokens per batch
    batch_size = max(1, int(target_tokens_per_batch / gradient_accumulation_step / avg_tokens))
    
    # CRITICAL: Cap batch size to prevent memory issues
    # For sequences ~30 tokens average, max batch size should be much smaller
    #max_batch_size = min(128, max(8, int(4000 / avg_tokens)))  # Adaptive max based on sequence length
    #batch_size = min(batch_size, max_batch_size)
    
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
