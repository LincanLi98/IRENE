import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GSA_Encoder import GSAEncoder


class GraphMAE(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=3, heads=4, mask_ratio=0.15):
        super().__init__()
        self.encoder = GSAEncoder(input_dim, embed_dim, num_layers, heads)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, input_dim)
        )
        self.mask_ratio = mask_ratio

    def random_mask(self, x):
        """
        对输入 batch x 做随机掩蔽
        x: [B, N, D]
        返回: masked_x, mask indices
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # 每个节点随机值
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_keep, mask, ids_keep, ids_mask, ids_restore

    def forward(self, x, A, A_phi):
        """
        x: [B, N, D] 输入 EEG 特征
        A, A_phi: 图结构 [B, N, N]
        """
        B, N, D = x.shape
        x_masked, mask, ids_keep, ids_mask, ids_restore = self.random_mask(x)

        # 构造掩蔽 token 填充到缺失位置
        x_full = x.clone()
        mask_token = self.mask_token.expand(B, ids_mask.size(1), -1)
        x_full.scatter_(1, ids_mask.unsqueeze(-1).expand(-1, -1, D), mask_token)

        # 编码器：带掩蔽输入的图结构编码
        encoded = self.encoder(x_full, A, A_phi)  # [B, N, embed_dim]

        # 解码器：恢复原始特征
        pred = self.decoder(encoded)  # [B, N, D]
        loss = (pred - x).pow(2).mean(dim=-1)  # [B, N]
        loss = (loss * mask).sum() / mask.sum()  # 只在掩蔽位置计算

        return pred, loss, mask
