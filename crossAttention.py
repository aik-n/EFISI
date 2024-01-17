import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from torch.nn import MultiheadAttention
import numpy as np
from sparsemax import Sparsemax

def sparsemax(z):
    z = z-max(z)
    zsort = sorted(z, reverse=True) #升序
    out = len(z)
    sum=0
    for k in range(0,len(z)):
        sum+=zsort[k]
        value = (k+1)*zsort[k]
        if(value<=sum-1):
            out=k
            sum=sum-zsort[k]-1
            break
    threshold = np.array(sum/out)
    return np.maximum(z-threshold, 0)

class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.sparsemax = Sparsemax(dim=-1)


    def forward(self, context, img, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        context.cuda()
        img.cuda()
        if pad_mask is not None:
            pad_mask.cuda()

        b, c, w = context.shape

        # x = self.proj_in(context)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = context.permute(0, 2, 1).contiguous()
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(context)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(img)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(img)

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            pad_mask = pad_mask.permute(0, 1, 3, 2).contiguous()
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        # att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.sparsemax(att_weights)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]

        # print(out.shape)

        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        # out = self.proj_out(out)   # [batch_size, c, h, w]

        return out, att_weights
