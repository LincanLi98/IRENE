import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GSA_Encoder import GSAEncoder

class GraphMAE(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=3, heads=4, mask_ratio=0.15):
        super().__init__()
        self.encoder = GSAEncoder(input_dim, embed_dim, num_layers, heads)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, input_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, input_dim)
        )
        self.mask_ratio = mask_ratio

    def random_mask(self, x):
        """
        Randomly mask (channel, clip) pairs in x.
        x: [B, N, C, D]
        Returns: masked_x, mask matrix
        """
        B, N, C, D = x.shape
        total = N * C
        num_mask = int(total * self.mask_ratio)

        # For each sample in batch, randomly select mask positions on (N, C)
        mask = torch.zeros((B, N, C), dtype=torch.bool, device=x.device)
        for b in range(B):
            flat_idx = torch.randperm(total)[:num_mask]
            idx_n, idx_c = flat_idx // C, flat_idx % C
            mask[b, idx_n, idx_c] = True

            # Ensure no channel is fully masked across all clips
            for n in range(N):
                if mask[b, n].all():
                    # Randomly unmask one clip for this channel
                    keep_c = torch.randint(C, (1,))
                    mask[b, n, keep_c] = False

        x_masked = x.clone()
        # Fill masked positions with mask token
        mask_token = self.mask_token.expand(B, N, C, D)
        x_masked[mask] = mask_token[mask]

        return x_masked, mask

    def forward(self, x, A=None, A_phi=None):
        """
        x: [B, N, C, D]
        A, A_phi: Graph structure [B, N, N], optional for GSAEncoder
        """
        B, N, C, D = x.shape
        x_masked, mask = self.random_mask(x)  # mask: [B, N, C]
        # Encoder: process masked x
        encoded = self.encoder(x_masked, A, A_phi)  # [B, N, C, embed_dim]
        # Decoder: reconstruct original features
        pred = self.decoder(encoded)  # [B, N, C, D]
        # Only calculate reconstruction loss on masked positions
        loss = ((pred - x) ** 2).mean(dim=-1)  # [B, N, C]
        loss = (loss * mask).sum() / mask.sum().clamp_min(1)
        return pred, loss, mask
