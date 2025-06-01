import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        A_hat = A + torch.eye(A.size(0), device=A.device)  # 加入自环
        D_inv_sqrt = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        return self.linear(A_norm @ X)


class GSA_Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

        self.gamma = nn.Parameter(torch.tensor(1.0))  # 可学习结构门控强度

    def forward(self, x, A_phi):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.heads, -1).transpose(1, 2), qkv)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if A_phi is not None:
            A_phi = A_phi.unsqueeze(1)  # [B, 1, N, N]
            attn = attn + self.gamma * A_phi

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


class GSAEncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.gc1 = GCNLayer(dim, dim)
        self.gc2 = GCNLayer(dim, dim)
        self.norm1 = nn.LayerNorm(dim)

        self.attn = GSA_Attention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, A, A_phi):
        # Graph Convolution Block
        h = self.gc1(x, A)
        h = F.relu(h)
        h = self.gc2(h, A)
        x = self.norm1(x + h)

        # Structure-aware Attention Block
        h = self.attn(x, A_phi)
        x = self.norm2(x + h)

        # FFN Block
        h = self.ffn(x)
        x = self.norm3(x + h)
        return x


class GSAEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=3, heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.layers = nn.ModuleList([
            GSAEncoderBlock(embed_dim, heads=heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, A, A_phi):
        """
        x: [B, N, input_dim] EEG特征输入
        A: [B, N, N] 稠密邻接矩阵
        A_phi: [B, N, N] 边置信度
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, A, A_phi)
        return x
