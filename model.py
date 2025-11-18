import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *

class Transformer(nn.Module):
    def __init__(self, n_encoder_layer, n_decoder_layer, embed_size, in_dim, out_dim, dropout,
                 encoder_args, decoder_args):
        # in_dim, out_dim : source vocab size, target vocab size
        super(Transformer, self).__init__()

        self.src_embed = nn.Embedding(in_dim, embed_size)
        self.trg_embed = nn.Embedding(out_dim, embed_size)
        self.positional = PositionalEncoding(embed_size, 5000, 'cpu')  # Added missing parameters

        self.src_dropout = nn.Dropout(dropout)
        self.trg_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(n_encoder_layer, embed_size, encoder_args)
        self.decoder = Decoder(n_decoder_layer, embed_size, decoder_args)

        self.out_linear = nn.Linear(embed_size, out_dim)

    def forward(self, source, target, src_mask=None, trg_mask=None, mode='train'):
        # source shape : (batch_size, src_seq_len)
        # target shape : (batch_size, trg_seq_len)
        # src_mask shape : (batch_size, src_seq_len) - True for real tokens, False for padding
        # trg_mask shape : (batch_size, trg_seq_len) - True for real tokens, False for padding
        batch_size, _ = source.shape

        # embedding
        source = self.src_embed(source)
        target = self.trg_embed(target)

        # positional encoding
        source = self.positional(source)
        target = self.positional(target)

        # dropout
        source = self.src_dropout(source)
        target = self.trg_dropout(target)

        encoder_output = self.encoder(source, src_mask) # (batch_size, src_seq_len, embed_dim)
        decoder_output = self.decoder(target, encoder_output, src_mask, trg_mask)

        # linear
        output = self.out_linear(decoder_output) # (batch_size, trg_seq_len, out_dim)
        
        # Apply softmax only during inference
        if mode == 'inference':
            output = F.softmax(output, dim=-1)
        
        return output


class Encoder(nn.Module):
    def __init__(self, n_encoder_layer, embed_dim, encoder_args):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_encoder_layer):
            self.layers.append(EncoderLayer(embed_dim, encoder_args))

    def forward(self, input, mask=None):
        # input shape (batch_size, src_seq_len, embed_dim)
        # mask shape (batch_size, src_seq_len) - True for real tokens, False for padding
        output = input
        for layer in self.layers:
            output = layer.forward(output, mask)
        return output # (batch_size, seq_len, embed_dim)
        
class Decoder(nn.Module):
    def __init__(self, n_decoder_layer, embed_dim, decoder_args):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_decoder_layer):
            self.layers.append(DecoderLayer(embed_dim, decoder_args))

    def forward(self, target, encoder_input, src_mask=None, trg_mask=None):
        # target shape (batch_size, seq_len, embed_dim)
        # src_mask shape (batch_size, src_seq_len) - encoder padding mask
        # trg_mask shape (batch_size, trg_seq_len) - decoder padding mask

        output = target
        for layer in self.layers:
            # forward
            output = layer.forward(output, encoder_input, src_mask, trg_mask)
        return output
