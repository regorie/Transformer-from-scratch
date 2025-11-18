import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, encoder_args):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiheadAttention(embed_dim, encoder_args['self_attention'])
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(encoder_args['dropout'])
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(encoder_args['dropout'])

    def forward(self, input, mask=None):
        # input shape (batch_size, seq_len, embed_dim)
        # mask shape (batch_size, seq_len) - True for real tokens, False for padding
        # output shape (batch_size, seq_len, embed_dim)
        residual = input.clone()

        # Apply dropout to sub-layer output BEFORE adding residual and normalizing
        output = self.attention(input, input, input, mask)
        output = self.dropout1(output)  # Dropout BEFORE residual connection
        output = self.layer_norm1(output + residual)

        residual = output.clone()

        # Apply dropout to sub-layer output BEFORE adding residual and normalizing
        output = self.ffn(output)
        output = self.dropout2(output)  # Dropout BEFORE residual connection
        output = self.layer_norm2(output + residual)

        return output

class DecoderLayer(nn.Module):
    #def __init__(self, embed_dim, n_head, d_k, d_v, device, max_length):
    def __init__(self, embed_dim, decoder_args):
        super(DecoderLayer, self).__init__()

        self.masked_attention = MultiheadAttention(embed_dim, decoder_args['self_attention'])
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(decoder_args['dropout'])
        self.attention = MultiheadAttention(embed_dim, decoder_args['attention'])
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(decoder_args['dropout'])
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.layer_norm3 = LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(decoder_args['dropout'])

    def forward(self, input, encoder_output, src_mask=None, trg_mask=None):
        # input shape (batch_size, trg_seq_len, embed_dim)
        # encoder_output shape (batch_size, src_seq_len, embed_dim)
        # src_mask shape (batch_size, src_seq_len) - encoder padding mask
        # trg_mask shape (batch_size, trg_seq_len) - decoder padding mask

        residual = input.clone()
        # Apply dropout to sub-layer output BEFORE adding residual and normalizing
        self_attention = self.masked_attention(input, input, input, trg_mask, causal=True)
        self_attention = self.dropout1(self_attention)  # Dropout BEFORE residual connection
        self_attention = self.layer_norm1(self_attention + residual)

        residual = self_attention.clone()
        # Apply dropout to sub-layer output BEFORE adding residual and normalizing
        output = self.attention(self_attention, encoder_output, encoder_output, src_mask)
        output = self.dropout2(output)  # Dropout BEFORE residual connection
        output = self.layer_norm2(output + residual)

        residual = output.clone()
        # Apply dropout to sub-layer output BEFORE adding residual and normalizing
        output = self.ffn(output)
        output = self.dropout3(output)  # Dropout BEFORE residual connection
        output = self.layer_norm3(output + residual)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, device):
        super(PositionalEncoding, self).__init__()

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, embed_dim, step=2, dtype=torch.float)

        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / embed_dim)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i[:embed_dim//2] / embed_dim)))
        
        self.register_buffer('pos_encoding', encoding)

    def forward(self, x):
        # x shape : (batch_size, seq_length, embed_dim)
        seq_len = x.shape[1]
        x = torch.add(x, self.pos_encoding[:seq_len,:])
        return x

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-12):
        super(LayerNorm, self).__init__()

        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        # shape (batch_size, seq_len, embed_dim)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class MultiheadAttention(nn.Module):
    #def __init__(self, embed_dim, n_head, d_k, d_v, device, max_length=None, mask=False):
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

        if max_length and args.get('mask', False):
            mask = torch.triu(torch.ones(max_length, max_length), diagonal=1)
            mask = mask.masked_fill(mask==1, float('-inf'))
            self.register_buffer('mask', mask)
        else:
            self.mask = None

    def forward(self, query, key, value, padding_mask=None, causal=False):
        # query shape : (batch_size, q_seq_len, embed_dim)
        # key, value shape : (batch_size, seq_len, embed_dim)
        # padding_mask shape : (batch_size, seq_len) - True for real tokens, False for padding
        # output (batch_size, q_seq_len, embed_dim)

        batch_size, q_seq_len, embed_dim = query.shape
        batch_size, seq_len, embed_dim = key.shape

        query = self.W_q(query) # (batch_size, q_seq_len, d_k*n_head)
        key = self.W_k(key) # (batch_size, seq_len, d_k*n_head)
        value = self.W_v(value) # (batch_size, seq_len, d_v*n_head)
        
        # Reshape for multi-head attention
        query = query.reshape(batch_size, q_seq_len, self.n_head, self.d_k) # (batch_size, q_seq_len, n_head, d_k)
        key = key.reshape(batch_size, seq_len, self.n_head, self.d_k) # (batch_size, seq_len, n_head, d_k)
        value = value.reshape(batch_size, seq_len, self.n_head, self.d_v) # (batch_size, seq_len, n_head, d_v)

        # Transpose to (batch_size, n_head, q_seq_len, d_k) and (batch_size, n_head, seq_len, d_k)
        query = query.transpose(1, 2)  # (batch_size, n_head, q_seq_len, d_k)
        key = key.transpose(1, 2)      # (batch_size, n_head, seq_len, d_k)
        value = value.transpose(1, 2)  # (batch_size, n_head, seq_len, d_v)
        
        # Compute attention scores
        attention = torch.matmul(query, key.transpose(-2, -1)) # (batch_size, n_head, q_seq_len, seq_len)
        
        # Apply causal mask if needed (for decoder self-attention)
        if causal or self.mask is not None:
            # Check if we can use pre-computed mask
            if (self.mask is not None and 
                q_seq_len <= self.mask.size(0) and 
                seq_len <= self.mask.size(1)):
                # Use pre-computed mask if sequences fit
                mask_slice = self.mask[:q_seq_len, :seq_len]
            else:
                # Create dynamic causal mask for longer sequences or when self.mask is None
                mask_slice = torch.triu(torch.ones(q_seq_len, seq_len, device=attention.device), diagonal=1)
                mask_slice = mask_slice.masked_fill(mask_slice == 1, float('-inf'))
            
            mask_expanded = mask_slice.unsqueeze(0).unsqueeze(0)
            mask_expanded = mask_expanded.expand(batch_size, self.n_head, q_seq_len, seq_len)
            attention = attention + mask_expanded
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Convert padding mask to attention mask
            # padding_mask: (batch_size, seq_len) - True for real tokens, False for padding
            # We need to mask out the padded positions in the key/value sequence
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            padding_mask_expanded = padding_mask_expanded.expand(batch_size, self.n_head, q_seq_len, seq_len)
            attention = attention.masked_fill(~padding_mask_expanded, float('-inf'))

        attention = F.softmax(attention / math.sqrt(self.d_k), dim=-1) # (batch_size, n_head, q_seq_len, seq_len)
        
        # Handle case where all attention weights are -inf (all padded)
        attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply attention to values
        output = torch.matmul(attention, value) # (batch_size, n_head, q_seq_len, d_v)
        output = output.transpose(1, 2) # (batch_size, q_seq_len, n_head, d_v)

        # concat and linear
        output = output.reshape(batch_size, q_seq_len, self.n_head * self.d_v)
        output = self.W_concat(output) # (batch_size, q_seq_len, embed_dim)

        return output
