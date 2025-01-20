##input embeddings
import torch 
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
        #return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  

class PositionalEncodring(nn.Module):
    def __init__(self, d_model:int, seq_len:int,dropout:float)-> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) ## To avoid overfittling
        ## Create a positional encoding
        self.pe = torch.zeros(seq_len, d_model)
        ## Creat a vector of shape 
        position = torch.arange(0, seq_len).unsqueeze(1) ## (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  ## To have the numerical stability we will take log and then applied exponetial
        ## apply the sin to even position
        self.pe[:, 0::2] = torch.sin(position.float() * div_term)
        ## apply the cos to odd position
        self.pe[:, 1::2] = torch.cos(position.float() * div_term)
      
        self.pe = self.pe.unsqueeze(0)
        self.register_buffer('pe', self.pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1),:]).requires_grad(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.beta = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0 , "d_model should be divisible by h"
        self.d_k = self.d_model // self.h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v= nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        query  = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # Split the embedding into self.h heads
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)