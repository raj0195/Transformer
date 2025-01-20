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
        