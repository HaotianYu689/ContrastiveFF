import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Patchify 工具 ----------
def img_to_patch(x, patch_size):
    """B,3,H,W → B,N,(3*P*P)"""
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dim not divisible by patch size"
    x = x.reshape(B, C,
                  H // patch_size, patch_size,
                  W // patch_size, patch_size)          # B,C,Hc,P,Wc,P
    x = x.permute(0, 2, 4, 1, 3, 5).flatten(1, 2)        # B,N,C,P,P
    x = x.flatten(2, 4)                                  # B,N,3*P*P
    return x                                             # N = (H/P)*(W/P)

# ---------- 子模块 ----------
class PatchingLayer(nn.Module):
    """把 B×3×32×32 → B×N×patch_dim，并可选拼接 class token"""
    def __init__(self, opt):
        super().__init__()
        self.P = opt.patch_size
        self.num_class = opt.num_class
        self.label_bank = None

    def set_label_rep(self, x):
        """随机初始化 class token，维度与 patch 吻合"""
        patch_dim = x.size(-1)
        bank = torch.zeros(self.num_class, 1, patch_dim, device=x.device)
        for c in range(self.num_class):
            idx = torch.randint(0, patch_dim, (int(patch_dim*0.2),))
            bank[c, 0, idx] = x.max()
        self.label_bank = bank     # num_class ×1×patch_dim

    def forward(self, x, y=None):
        x = img_to_patch(x, self.P)           # B,N,patch_dim
        if y is not None:
            if self.label_bank is None:
                self.set_label_rep(x)
            cls_tok = self.label_bank[y]      # B,1,patch_dim
            x = torch.cat([cls_tok, x], dim=1)
        return x                              # B , N(+1) , patch_dim

class PositionalEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, opt.num_patches, opt.E))
    def forward(self, x):
        return x + self.pos[:, :x.size(1)]     # 兼容是否加了 cls token

class ViTEncoder(nn.Module):
    def __init__(self, E, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(E)
        self.attn  = nn.MultiheadAttention(E, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(E)
        self.mlp   = nn.Sequential(
            nn.Linear(E, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, E),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + h                          # residual 1
        x = x + self.mlp(self.norm2(x))    # residual 2
        return x

# ---------- ViT 主体 ----------
class ViT(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.patch_dim = 3 * (opt.patch_size ** 2)    # 48 (when P=4)
        self.E         = opt.E

        self.patching_layer = PatchingLayer(opt)

        # 第一块：patch → embedding
        first_block = nn.Sequential(
            nn.Linear(self.patch_dim, self.E),
            nn.ReLU(),
            PositionalEncoder(opt),
            ViTEncoder(self.E, self.E*2, opt.H)
        )

        # 后续 Transformer blocks
        other_blocks = [ViTEncoder(self.E, self.E*2, opt.H) for _ in range(1, opt.L)]

        self.layers = nn.ModuleList([first_block] + other_blocks)

        # classification head
        self.head = nn.Sequential(
            nn.LayerNorm(self.E),
            nn.Linear(self.E, opt.num_class)
        )

    def forward(self, x, y=None):
        # --------- 调试输出 ---------
        # print("x0 :", x.shape)    # e.g. (B,3,32,32)
        # ---------------------------

        x = self.patching_layer(x, y)       # B,N,patch_dim
        for block in self.layers:
            x = block(x)                    # B,N,E

        x = x.mean(1)                       # global average
        return self.head(x)                 # B,num_class

