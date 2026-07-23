#!/home/michal/miniforge3/envs/TFS/bin/python

'''Implementation of https://arxiv.org/pdf/1706.03762'''

import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    '''Each token is represented as a fixed-size, high-dimensional, vector (d=512). (p. 5 sec. 3.4)'''

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):

        return self.embedding(x) * math.sqrt(self.d_model) # Multiplication by sqrt(d_model) following the original paper 


class PositionalEncoding(nn.Module):
    '''To each token, we add a learned position embedding of the same dimension. (p. 6 sec. 3.5)'''

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)


