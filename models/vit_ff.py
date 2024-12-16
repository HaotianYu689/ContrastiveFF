import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PatchingLayer(nn.Module):
    
    def __init__(self,opt):
        super().__init__()
        self.patch_size = opt.patch_size
        self.labels     = torch.randn(10,self.patch_size * self.patch_size*3).cuda()

    def forward(self, x, y):

        B, C, H, W = x.shape

        # reshape to B x C x H_count x H_patch x W_count x W_patch
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1,2)
        x = x.flatten(2, 4)
        x = torch.cat([self.labels[y], x], dim=1)
        return x
    

class PositionalEncoder(nn.Module):
    
    def __init__(self,opt):
        super().__init__()
        # Learnable parameters for position embedding
        self.pos_embed   = nn.Parameter(torch.randn((1, opt.num_patches, opt.E)))


    def forward(self, x):
        return x + self.pos_embed

class ViTEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.attn  = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden_dim, input_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.norm1(x)
        out, _ = self.attn(out, out, out)

        # First residual connection
        resid = x + out

        # Pass through MLP layer
        out = self.norm2(resid)
        out = self.act(self.fc1(out))
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)

        # Second residual connection
        out = out + resid

        return out


class ViT(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.patching_layer = PatchingLayer(opt)
        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(nn.Sequential(
          nn.Linear(3*(opt.patch_size**2), opt.E),
          nn.ReLU(),
          PositionalEncoder(opt),
          ViTEncoder(opt.E, opt.E*2, opt.H)
        ))
        
        # Another layers
        self.layers.extend([ViTEncoder(opt.E, opt.E*2, opt.H) for _ in range(1,opt.L)])
            
        # Classification head
        self.classifier_head = nn.Sequential(nn.LayerNorm(opt.E),
                                             nn.Linear(opt.E, opt.num_class))

    def forward(self, x):
        # Encoding
        for layer in self.layers:
            x = layer(x)
        
        # AVG pooling
        x = x.mean(1)

        # Pass through classification head
        x = self.classifier_head(x)
        return x
