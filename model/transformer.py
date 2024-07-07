import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, token_len, d_model=512):
        super(PositionalEmbedding, self).__init__()
        self.pos_encoding = torch.zeros(token_len, d_model)

        position = torch.arange(0, token_len, dtype=torch.float).unsqueeze(1)
        # 避免不同维度上的正弦和余弦波形频率相同，从而导致编码信息的冗余
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 填充位置编码矩阵
        self.pos_encoding[:, 0::2] = torch.sin(position * div_term)
        self.pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        # 将位置编码矩阵的形状调整为 (1, max_len, d_model)
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.pos_encoding
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, qkv_dim):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_heads
        self.embed_dim = embed_dim
        self.qkv_dim = qkv_dim

        # key, query, value projections for all heads, but in a batch
        self._attn = nn.Linear(embed_dim, 3 * embed_dim)
        # output projection
        self._proj = nn.Linear(embed_dim, embed_dim)

    
    def forward(self, x):
        BATCH, SEQ_LEN, DIM = x.size()

        # calculate Q, K, V in a bath
        qkv = self._attn(x)
        q,k,v = qkv.split(self.qkv_dim, dim=2) # (B, T, DIM)
        q = q.view(BATCH, SEQ_LEN, self.num_head, DIM // self.num_head).transpose(1, 2) # (B, nh, T, DIM/nh)
        k = k.view(BATCH, SEQ_LEN, self.num_head, DIM // self.num_head).transpose(1, 2) # (B, nh, T, DIM/nh)
        v = v.view(BATCH, SEQ_LEN, self.num_head, DIM // self.num_head).transpose(1, 2) # (B, nh, T, DIM/nh)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # flash self attention
        y = y.transpose(1, 2).contiguous().view(BATCH, SEQ_LEN, DIM) # re-assemble all head outputs side by side
        # output projection
        y = self._proj(y)
        return y
    

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super(MLP, self).__init__()
        self.c_fc    = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, num_heads, embed_dim, qkv_dim):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(num_heads=num_heads, embed_dim=embed_dim, qkv_dim=qkv_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class Transformer(nn.Module):
    def __init__(self, token_len, num_layer=8, num_heads=8, embed_dim=512, qkv_dim=512, action_bin_size=256):
        super(Transformer, self).__init__()
        self.pos_embed =  PositionalEmbedding(token_len=token_len, d_model= embed_dim)
        self.blocks = [Block(num_heads, embed_dim, qkv_dim) for _ in range(num_layer)]
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, action_bin_size, bias=False)

    
    def forward(self, x):
        """
        x: batch x num_tokens x 512
        """
        x = self.pos_embed(x) # batch x num_tokens x 512

        for block in self.blocks:
            x = block(x) # batch x num_tokens x 512

        x = self.ln_f(x) # batch x num_tokens x 512
        
        x = self.lm_head(x) # batch x num_tokens x action_bin_size
        return x
    



# t = Transformer()
# input = torch.randn((1,48,512))
# print(t(input).shape)
