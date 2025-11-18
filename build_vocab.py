from collections import Counter
import pickle
import torch

PAD=0
UNK=1
BOS=2
EOS=3

def build_vocab(file_paths, vocab_size=None, min_freq=1):
    """Build vocabulary from BPE-processed files"""
    token_counter = Counter()

    # Count all tokens 
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                token_counter.update(tokens)

    # special tokens
    special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']

    # create vocab, add special tokens first
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    # add most frequent tokens
    most_common = token_counter.most_common(vocab_size - len(special_tokens) if vocab_size else None)
    for token, freq in most_common:
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    # create reverse mapping
    idx_to_token = {idx: token for token, idx in vocab.items()}

    return vocab, idx_to_token, token_counter


def load_vocab(vocab_file):
    """Load vocabulary from pickle file"""
    with open(vocab_file, 'rb') as f:
        vocab, idx2tok = pickle.load(f)
    return vocab, idx2tok


if __name__=="__main__":

    # build vocabulary
    vocab_en, idx2tok_en, counter_en = build_vocab(['./datasets/wmt14_en_de/train.en', './datasets/wmt14_en_de/valid.en'])
    vocab_de, idx2tok_de, counter_de = build_vocab(['./datasets/wmt14_en_de/train.de', './datasets/wmt14_en_de/valid.de'])

    print(f"English vocab size: {len(vocab_en)}")
    print(f"German vocab size: {len(vocab_de)}")

    # Save vocabularies
    with open('vocab_en.pkl', 'wb') as f:
        pickle.dump((vocab_en, idx2tok_en), f)
        
    with open('vocab_de.pkl', 'wb') as f:
        pickle.dump((vocab_de, idx2tok_de), f)

