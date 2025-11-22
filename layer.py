import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, encoder_args):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiheadAttention(embed_dim, encoder_args['self_attention'])
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(encoder_args['dropout'])
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout2 = nn.Dropout(encoder_args['dropout'])

    def forward(self, input, mask=None):
        # input shape (batch_size, seq_len, embed_dim)
        # mask shape (batch_size, seq_len) - True for real tokens, False for padding
        # output shape (batch_size, seq_len, embed_dim)
        residual = input.clone()

        output = self.attention(input, input, input, mask, mask)
        output = self.dropout1(output)
        output = self.layer_norm1(output + residual)

        residual = output.clone()

        output = self.ffn(output)
        output = self.dropout2(output)
        output = self.layer_norm2(output + residual)

        return output

class DecoderLayer(nn.Module):
    #def __init__(self, embed_dim, n_head, d_k, d_v, device, max_length):
    def __init__(self, embed_dim, decoder_args):
        super(DecoderLayer, self).__init__()

        self.masked_attention = MultiheadAttention(embed_dim, decoder_args['self_attention'])
        self.layer_norm1 = nn.LayerNorm(embed_dim,eps=1e-6)
        self.dropout1 = nn.Dropout(decoder_args['dropout'])
        self.attention = MultiheadAttention(embed_dim, decoder_args['attention'])
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout2 = nn.Dropout(decoder_args['dropout'])
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm3 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout3 = nn.Dropout(decoder_args['dropout'])

    def forward(self, input, encoder_output, src_mask=None, trg_mask=None):
        # input shape (batch_size, trg_seq_len, embed_dim)
        # encoder_output shape (batch_size, src_seq_len, embed_dim)
        # src_mask shape (batch_size, src_seq_len) - encoder padding mask
        # trg_mask shape (batch_size, trg_seq_len) - decoder padding mask

        residual = input.clone()
        self_attention = self.masked_attention(input, input, input, trg_mask, trg_mask, masked=False)
        self_attention = self.dropout1(self_attention)
        self_attention = self.layer_norm1(self_attention + residual)

        residual = self_attention.clone()
        output = self.attention(self_attention, encoder_output, encoder_output, None, src_mask)
        output = self.dropout2(output)
        output = self.layer_norm2(output + residual)

        residual = output.clone()
        output = self.ffn(output)
        output = self.dropout3(output)
        output = self.layer_norm3(output + residual)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, device='cpu'):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embed_dim, step=2, dtype=torch.float)

        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / embed_dim)))
        
        self.register_buffer('pos_encoding', encoding)

    def forward(self, x):
        # x shape : (batch_size, seq_length, embed_dim)
        seq_len = x.shape[1]
        # pos_encoding is automatically on the same device as x due to register_buffer
        return x + self.pos_encoding[:seq_len, :].unsqueeze(0)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, args):
        super(MultiheadAttention, self).__init__()
        d_k = args['d_k']
        d_v = args['d_v']
        n_head = args['n_head']
        max_length = args['max_length']

        self.W_q = nn.Linear(embed_dim, d_k*n_head)
        self.W_k = nn.Linear(embed_dim, d_k*n_head)
        self.W_v = nn.Linear(embed_dim, d_v*n_head)
        self.W_concat = nn.Linear(d_v*n_head, embed_dim)

        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.attention = None


    def forward(self, query, key, value, q_padding_mask=None, v_padding_mask=None, masked=False):
        # query shape : (batch_size, q_seq_len, embed_dim)
        # key, value shape : (batch_size, seq_len, embed_dim)
        # q_padding_mask shape : (batch_size, q_seq_len) = True for real tokens, False for padding
        # v_padding_mask shape : (batch_size, seq_len) = True for real tokens, False for padding
        # output (batch_size, q_seq_len, embed_dim)

        batch_size, q_seq_len, embed_dim = query.shape
        batch_size, seq_len, embed_dim = value.shape

        query = self.W_q(query) # (batch_size, q_seq_len, d_k*n_head)
        key = self.W_k(key) # (batch_size, seq_len, d_k*n_head)
        value = self.W_v(value) # (batch_size, seq_len, d_v*n_head)
        
        # Split for multi-head attention
        query = query.reshape(batch_size, q_seq_len, self.n_head, self.d_k) # (batch_size, q_seq_len, n_head, d_k)
        key = key.reshape(batch_size, seq_len, self.n_head, self.d_k) # (batch_size, seq_len, n_head, d_k)
        value = value.reshape(batch_size, seq_len, self.n_head, self.d_v) # (batch_size, seq_len, n_head, d_v)

        # Transpose to (batch_size, n_head, q_seq_len, d_k) and (batch_size, n_head, seq_len, d_k)
        query = query.transpose(1, 2)  # (batch_size, n_head, q_seq_len, d_k)
        key = key.transpose(1, 2)      # (batch_size, n_head, seq_len, d_k)
        value = value.transpose(1, 2)  # (batch_size, n_head, seq_len, d_v)
        
        # Compute attention scores
        attention = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k) # (batch_size, n_head, q_seq_len, seq_len)
        #if torch.isinf(attention).any():
        #    print("inf value in attention score before masking")
        # Apply causal mask for decoder self-attention (before padding mask)
        if masked:
            causal_mask = torch.triu(torch.ones(q_seq_len, seq_len, device=attention.device), diagonal=1).bool()
            attention = attention.masked_fill(causal_mask, float('-inf'))
        #if torch.isinf(attention).any():
        #    print("inf value in attention score after causal masking")

        # Apply padding mask
        if v_padding_mask is not None:
            # v_padding_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = v_padding_mask.unsqueeze(1).unsqueeze(1)
            attention = attention.masked_fill(~mask, float('-inf'))

        # Softmax
        #if attention.max() > 50:
        #    print("attention scores too high")
        #if torch.isinf(attention).any():
        #    print("inf value in attention score before softmax")
        attention = F.softmax(attention, dim=-1)
        #if torch.isnan(attention).any():
        #    print("nan values after softmax")

        self.attention = attention

        # Apply attention to values
        # (batch_size, n_head, q_seq_len, seq_len) * (batch_size, n_head, seq_len, d_v) -> (batch_size, n_head, q_seq_len, d_v)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2) # (batch_size, q_seq_len, n_head, d_v)

        # concat and linear
        output = output.reshape(batch_size, q_seq_len, self.n_head * self.d_v)
        output = self.W_concat(output) # (batch_size, q_seq_len, embed_dim)
        return output

