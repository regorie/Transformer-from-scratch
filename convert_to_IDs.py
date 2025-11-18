import pickle
import torch
from torch.utils.data import Dataset, DataLoader

def load_vocab(vocab_file):
    """Load vocabulary from pickle file"""
    with open(vocab_file, 'rb') as f:
        vocab, idx_to_token = pickle.load(f)
    return vocab, idx_to_token

def text_to_ids(text, vocab, add_special_tokens=True):
    """Convert BPE text to sequence of IDs"""
    tokens = text.strip().split()
    
    if add_special_tokens:
        tokens = ['<bos>'] + tokens + ['<eos>']
    
    # Convert to IDs, use <unk> for unknown tokens
    ids = []
    for token in tokens:
        ids.append(vocab.get(token, vocab['<unk>']))
    
    return ids

def ids_to_text(ids, idx_to_token):
    """Convert sequence of IDs back to text"""
    tokens = []
    for id_ in ids:
        if id_ in idx_to_token:
            token = idx_to_token[id_]
            if token not in ['<pad>', '<bos>', '<eos>']:
                tokens.append(token)
    
    # Join and clean up BPE
    text = ' '.join(tokens)
    # Remove BPE boundary markers and merge subwords
    text = text.replace('</w>', ' ').strip()
    # Handle multiple spaces
    text = ' '.join(text.split())
    
    return text

if __name__=="__main__":
    # load vocabulary
    vocab_en, idx2tok_en = load_vocab('vocab_en.pkl')
    vocab_de, idx_to_token_de = load_vocab('vocab_de.pkl')

    input_files = ['./datasets/wmt14_en_de/train.en', './datasets/wmt14_en_de/train.de',
                   './datasets/wmt14_en_de/valid.en', './datasets/wmt14_en_de/valid.de']
    output_files = ['./datasets/wmt14_en_de/train_ids.en', './datasets/wmt14_en_de/train_ids.de',
                    './datasets/wmt14_en_de/valid_ids.en', './datasets/wmt14_en_de/valid_ids.de']
    
    for in_file, out_file in zip(input_files, output_files):
        with open(in_file, 'r', encoding='utf-8') as fin, \
            open(out_file, 'w', encoding='utf-8') as fout:

            if 'en' in in_file:
                for line in fin:
                    ids = text_to_ids(line, vocab_en)
                    fout.write(' '.join(map(str, ids))+'\n')
            elif 'de' in in_file:
                for line in fin:
                    ids = text_to_ids(line, vocab_de)
                    fout.write(' '.join(map(str, ids))+'\n')