import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B, Head, F, F]
        if attn_mask:
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F,F]


class Similarity_matrix(nn.Module):
    ''' buliding similarity matrix by self-attention mechanism '''

    def __init__(self, num_heads=4, input_size=512, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = input_size
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix = self.attention(query, key, value, attn_mask)

        return matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
