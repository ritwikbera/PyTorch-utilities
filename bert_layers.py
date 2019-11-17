import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class PositionalEncoder(nn.Module):
    def __init__(self, input_embed_dim, max_seq_len = 80):
        super().__init__()
        self.input_embed_dim = input_embed_dim
        pe = torch.zeros(max_seq_len, input_embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, input_embed_dim, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/input_embed_dim)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/input_embed_dim)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.input_embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],requires_grad=False)
        return x

def attention(q, k, v, d_k, mask=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
    
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads=1, input_embed_dim, output_dim):
        super().__init__()
        
        self.input_embed_dim = input_embed_dim
        self.output_dim = output_dim
        self.d_k = output_dim // heads
        self.h = heads
        
        self.q_linear = nn.Linear(input_embed_dim, output_dim)
        self.v_linear = nn.Linear(input_embed_dim, output_dim)
        self.k_linear = nn.Linear(input_embed_dim, output_dim)
        self.out = nn.Linear(output_dim, input_embed_dim)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask) 
        print('Score size {}'.format(scores.size()))
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.output_dim)
        print('Concatenated scores size {}'.format(concat.size()))
        output = self.out(concat)
        return output

if __name__=='__main__':
    mhatt = MultiHeadAttention(2,5)
    q = torch.randn(1, 10, 2)
    k = torch.randn(1, 10, 2)
    v = torch.randn(1, 10, 2)
    out = mhatt(q,k,v)
    print(out.size())