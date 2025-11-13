import torch
import torch.utils.data as data
import torch.nn as nn
from tqdm import tqdm
from minbpe import BasicTokenizer
import os
import pickle

# tokens
PAD = 0
SOS = 1
EOS = 2

def load_data(src_data_path, trg_data_path, tokenizer, max_length, source_file, target_file):
    print("loading data...")

    # check if tokenized file exist
    if os.path.exists(source_file) and os.path.exists(target_file):
        print(f"Loading data from tokenized data from {source_file} and {target_file}")
        # load tokenized content
        with open(source_file, 'rb') as sf:
            src_content = pickle.load(sf)
        with open(target_file, 'rb') as tf:
            trg_content = pickle.load(tf)

        return src_content['src_sentences'], trg_content['trg_sentences'], src_content['avg_length']
    else:
        avg_length = 0
        count = 0
        filtered_count = 0
        with open(src_data_path, 'r', encoding='utf-8') as sf,\
            open(trg_data_path, 'r', encoding='utf-8') as tf:
            src_lines = sf.readlines()
            trg_lines = tf.readlines()

            src_sentences = []
            trg_sentences = []

            print("Loading and filtering data...")
            for src_line, trg_line in tqdm(zip(src_lines, trg_lines), total=len(src_lines)):
                if len(src_line) > 5 and len(trg_line) > 5:
                    src_tokens = tokenizer.encode(src_line.strip())
                    trg_tokens = tokenizer.encode(trg_line.strip())

                    # Filter out sequences that are too long
                    if len(src_tokens) <= max_length and len(trg_tokens) <= max_length:
                        avg_length += len(src_tokens)
                        avg_length += len(trg_tokens)

                        src_sentences.append(src_tokens)
                        trg_sentences.append(trg_tokens)
                        count += 2
                    else:
                        filtered_count += 1
                    
            avg_length /= count
            print("total sentence pairs: ", count/2)
            print("average length: ", avg_length)
            if filtered_count > 0:
                print(f"filtered out {filtered_count} sentences longer than {max_length} tokens")

            # save tokenized files
            src_content = {
                'src_sentences': src_sentences,
                'avg_length': avg_length
            }
            trg_content = {
                'trg_sentences':trg_sentences,
                'avg_length': avg_length
            }

            os.makedirs(os.path.dirname(source_file), exist_ok=True)
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            with open(source_file, 'wb') as sf:
                pickle.dump(src_content, sf)
            with open(target_file, 'wb') as tf:
                pickle.dump(trg_content, tf)

            return src_sentences, trg_sentences, avg_length

def load_tokenizer(file_path):
    tokenizer = BasicTokenizer()
    tokenizer.load(file_path)
    return tokenizer

def get_collate_fn():
    def collate_fn(batch):
        sources = []
        targets = []
        src_lengths = []

        for item in batch:
            sources.append(item['source'])
            targets.append(item['target'])
            src_lengths.append(len(item['source']))

        src_padded = nn.utils.rnn.pad_sequence(sources, padding_value=PAD, batch_first=True)
        target_padded = nn.utils.rnn.pad_sequence(targets, padding_value=PAD, batch_first=True)

        return {
            'source': src_padded,
            'target': target_padded,
            'src_lengths': torch.tensor(src_lengths, dtype=torch.long)
        }
    
    return collate_fn

def get_data_loader(dataset, batch_size, shuffle=False, drop_last=False):
    collate_fn = get_collate_fn()
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return data_loader

class TextDataset(data.Dataset):
    def __init__(self, src_sentences, trg_sentences):
        """
        src_sentences: list of sentences(list of word idices)
        trg_sentences: same as above
        src_vocab = (src_w2i, src_i2w)
        trg_vocab = same as above
        """
        #self.src_sentences = src_sentences
        #self.trg_sentences = trg_sentences

        #self.length = len(src_sentences)
        self.src_tensors = []
        self.trg_tensors = []

        for i in tqdm(range(len(trg_sentences))):
            trg_with_tokens = [SOS] + trg_sentences[i] + [EOS]
            self.src_tensors.append(torch.tensor(src_sentences[i]))
            self.trg_tensors.append(torch.tensor(trg_with_tokens))


    def __len__(self):
        return len(self.src_tensors)
    
    def __getitem__(self, idx):
        # 1. source: input to encoder
        # 2. target: target sequence
        #            use target[:-1] for decoder input
        #            use target[1:] for decoder output

        return { 
            'source': self.src_tensors[idx],
            'target': self.trg_tensors[idx]
        }