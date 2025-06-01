import torch
import torch.nn as nn
import torch.nn.functional as F
#from model.GSA_Encoder import GSAEncoder
from GSA_Encoder import GSAEncoder

class IRENEModel_classification(nn.Module):
    def __init__(self, args, num_classes, device):
        super().__init__()
        self.encoder = GSAEncoder(input_dim=args.input_dim, embed_dim=args.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(args.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, seq_lengths=None, adj=None):
        z = self.encoder(x, adj)
        out = self.mlp(z.mean(dim=1))  # 可改为 attention-pool
        return out, z
