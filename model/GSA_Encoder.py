import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchFrequencyEmbedding(nn.Module):
    """
    Projects the FFT frequency features to embedding space.
    """
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GCNLayer(nn.Module):
    """
    GCN with self-loop and symmetric normalization.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        # X: [B, N, D], A: [B, N, N]
        B, N, _ = X.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = A + I
        D_inv_sqrt = torch.pow(A_hat.sum(-1), -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag_embed(D_inv_sqrt)
        A_norm = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        X = torch.bmm(A_norm, X)
        return self.linear(X)

class GSA_Attention(nn.Module):
    """
    Graph Structure-Aware Multi-Head Attention.
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Structure prior strength

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
    """
    GCN x2 + Structure-aware Attention + FFN, each with residual and LayerNorm.
    """
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
    def __init__(self, input_dim, embed_dim, num_layers=3, heads=4, n_fft=200, hop_length=100, n_channels=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_channels = n_channels
        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=embed_dim, n_freq=n_fft // 2 + 1
        )
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            GSAEncoderBlock(embed_dim, heads=heads)
            for _ in range(num_layers)
        ])

    def stft(self, sample):
        # sample: [B, 1, T]
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x, A, A_phi):
        """
        x: [B, N, T] raw time series EEG (before FFT)
        A: [B, N, N]
        A_phi: [B, N, N]
        Output: [B, N*ts, embed_dim]
        """
        B, N, T = x.shape
        emb_seq = []
        for i in range(N):
            channel_spec_emb = self.stft(x[:, i : i + 1, :])   # [B, F, ts]
            channel_spec_emb = self.patch_embedding(channel_spec_emb)  # [B, ts, embed_dim]
            # Positional encoding
            channel_emb = self.positional_encoding(channel_spec_emb)
            emb_seq.append(channel_emb)
        # Concatenate all channels on the sequence dimension: [B, N*ts, embed_dim]
        emb = torch.cat(emb_seq, dim=1)
        # Feed into GCN + Structure-aware transformer blocks
        # Here we reshape emb to [B, N*ts, embed_dim] -> [B, N, ts, embed_dim]
        ts = emb_seq[0].shape[1]
        emb = emb.view(B, N, ts, -1)
        # Merge ts into N (flatten into [B, N*ts, embed_dim])
        x = emb.view(B, N * ts, -1)
        for layer in self.layers:
            x = layer(x, A, A_phi)
        return x
