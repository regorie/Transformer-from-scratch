import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *
from dataset import PAD

class Transformer(nn.Module):
    def __init__(self, n_encoder_layer, n_decoder_layer, embed_size, vocab_size, dropout,
                 encoder_args, decoder_args):
        # in_dim, out_dim : source vocab size, target vocab size
        super(Transformer, self).__init__()

        #self.src_embed = nn.Embedding(vocab_size, embed_size)
        #self.trg_embed = nn.Embedding(vocab_size, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD)
        self.positional = PositionalEncoding(embed_size, 5000)

        self.src_dropout = nn.Dropout(dropout)
        self.trg_dropout = nn.Dropout(dropout)

        self.encoder = Encoder(n_encoder_layer, embed_size, encoder_args)
        self.decoder = Decoder(n_decoder_layer, embed_size, decoder_args)

        self.out_linear = nn.Linear(embed_size, vocab_size, bias=False)
        self.out_linear.weight = self.embed.weight
        
        # Store embed_size for scaling
        self.embed_size = embed_size
        
        # Proper weight initialization
        self._init_weights()

    def forward(self, source, target, src_mask=None, trg_mask=None, mode='train', temperature=1.0):
        # source shape : (batch_size, src_seq_len)
        # target shape : (batch_size, trg_seq_len)
        # src_mask shape : (batch_size, src_seq_len) - True for real tokens, False for padding
        # trg_mask shape : (batch_size, trg_seq_len) - True for real tokens, False for padding
        # temperature : float to control output distribution sharpness
        batch_size, _ = source.shape

        source = self.embed(source) * self.embed_size**0.5
        target = self.embed(target) * self.embed_size**0.5

        # positional encoding
        source = self.positional(source)
        target = self.positional(target)

        # dropout
        source = self.src_dropout(source)
        target = self.trg_dropout(target)

        encoder_output = self.encoder(source, src_mask) # (batch_size, src_seq_len, embed_dim)
        decoder_output = self.decoder(target, encoder_output, src_mask, trg_mask)

        # linear
        output = self.out_linear(decoder_output)# (batch_size, trg_seq_len, out_dim)
        
        if temperature != 1.0:
            output = output / temperature
        
        return output

    def _init_weights(self):
        #for name, p in self.named_parameters():
        #    if p.dim() > 1:
        #        nn.init.xavier_uniform_(p)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def debug_forward(self, source, target, src_mask=None, trg_mask=None):
        """Debug version of forward pass with intermediate value checking"""
        print(f"🔍 Input source range: [{source.min().item():.3f}, {source.max().item():.3f}]")
        print(f"🔍 Input target range: [{target.min().item():.3f}, {target.max().item():.3f}]")
        
        # embedding with scaling
        source_emb = self.src_embed(source) * self.embed_size**0.5
        target_emb = self.trg_embed(target) * self.embed_size**0.5
        print(f"🔍 After embedding - source range: [{source_emb.min().item():.3f}, {source_emb.max().item():.3f}]")
        print(f"🔍 After embedding - target range: [{target_emb.min().item():.3f}, {target_emb.max().item():.3f}]")
        
        # positional encoding
        source_pos = self.positional(source_emb)
        target_pos = self.positional(target_emb)
        print(f"🔍 After positional - source range: [{source_pos.min().item():.3f}, {source_pos.max().item():.3f}]")
        print(f"🔍 After positional - target range: [{target_pos.min().item():.3f}, {target_pos.max().item():.3f}]")
        
        # Check for NaN/Inf
        if torch.isnan(source_pos).any() or torch.isinf(source_pos).any():
            print("⚠️ NaN/Inf detected in source after positional encoding!")
        if torch.isnan(target_pos).any() or torch.isinf(target_pos).any():
            print("⚠️ NaN/Inf detected in target after positional encoding!")
            
        return source_pos, target_pos


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
