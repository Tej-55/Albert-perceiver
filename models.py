"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import split_last, merge_last


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = 30000 # Size of Vocabulary
    hidden: int = 384 # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = 640 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding: int = 64 # Factorized embedding parameterization

    n_layers: int = 6 # Numher of Hidden Layers
    n_heads: int = 384//32 # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 256 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    M: int = 256
    C: int = 384
    N: int = 128
    D: int = 384
    cross_heads: int = 1
    latent_heads: int = 8
    cross_dim_head: int = 32
    latent_dim_head: int = 32
    ffw: int = 640
    process_layers: int = 12

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))



class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.tok_embed2 = nn.Linear(cfg.embedding, cfg.hidden)

        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.hidden) # segment(token type) embedding

        self.norm = nn.LayerNorm(cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        #return self.drop(self.norm(e))
        return self.norm(e)

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta  = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class QKVAttention_cross(nn.Module):
    """ QKV Attention """
    def __init__(self, cfg, heads):
        super().__init__()
        self.proj_q = nn.Linear(cfg.D, cfg.D)     #self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.C, cfg.D)
        self.proj_v = nn.Linear(cfg.C, cfg.D)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = heads

    def forward(self, x, latent, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(latent), self.proj_k(x), self.proj_v(x)
        #print(x.shape, q.shape)
        #print(latent.shape, v.shape)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(-2, -3)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        #print(scores.shape, v.shape)
        h = (scores @ v).transpose(1, 2).contiguous()
        #print(h.shape)
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class QKVAttention_self(nn.Module):
    """ QKV Attention """
    def __init__(self, cfg, heads):
        super().__init__()
        self.proj_q = nn.Linear(cfg.D, cfg.D)
        self.proj_k = nn.Linear(cfg.D, cfg.D)
        self.proj_v = nn.Linear(cfg.D, cfg.D)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(-2, -3)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class FeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.D, cfg.ffw)
        self.fc2 = nn.Linear(cfg.ffw, cfg.D)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with QKV Attention Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.latents = nn.Parameter(torch.randn(cfg.N, cfg.D))#[None, :,:]

        self.cross_attend_blocks = nn.ModuleList([
            QKVAttention_cross(cfg, heads = cfg.cross_heads), #PreNorm(cfg.C, QKVAttention(cfg, heads = cfg.cross_heads), context_dim = cfg.C),
            FeedForward(cfg) #PreNorm(cfg.D, FeedForward(cfg))
        ])
        get_latent_attn = lambda: QKVAttention_self(cfg, heads = cfg.latent_heads) # PreNorm(cfg.D, QKVAttention(cfg, heads = cfg.latent_heads))
        get_latent_ff = lambda: FeedForward(cfg) #PreNorm(cfg.D, FeedForward(cfg))
        #get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([
            QKVAttention_self(cfg, heads = cfg.latent_heads),
            FeedForward(cfg)
        ])
        #cache_args = {'_cache': weight_tie_layers}



        # To used parameter-sharing strategies
        self.n_layers = cfg.process_layers
        # self.attn = QKVAttention(cfg)
        # self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = nn.LayerNorm(cfg.D)
        # self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = nn.LayerNorm(cfg.D)
        self.norm3 = nn.LayerNorm(cfg.D)
        self.norm4 = nn.LayerNorm(cfg.D)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = input_ids
        seg = token_type_ids
        mask = attention_mask
        x = self.embed(x, seg)

        cross_attn, cross_ff = self.cross_attend_blocks
        x = self.norm1(cross_attn(x, self.latents, mask = mask) + self.latents)

        x = self.norm2(cross_ff(x) + x)

        self_attn, self_ff = self.layers
        for _ in range(self.n_layers):

            x = self.norm3(self_attn(x, mask = None) + x)
            x = self.norm4(self_ff(x) + x)



        return x

