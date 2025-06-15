import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearScorer(nn.Module):
    """
    Bilinear scorer for InfoNCE term (I(A;Y)).
    """
    def __init__(self, a_dim, y_dim):
        super().__init__()
        self.scorer = nn.Bilinear(a_dim, y_dim, 1)

    def forward(self, A_feat, Y_feat):
        # A_feat: [B, d], Y_feat: [B, d2]
        return self.scorer(A_feat, Y_feat)

class DonskerCritic(nn.Module):
    def __init__(self, a_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(a_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, A_flat, Z):
        # A_flat: [B, d1], Z: [B, d2]
        in_vec = torch.cat([A_flat, Z], dim=-1)
        return self.net(in_vec)

class IRENE_Loss(nn.Module):
    """
    Total loss for IRENE model:
        - IB-Graph loss (self-expression + redundancy + info)
        - Temporal smoothness
        - Reconstruction
    """
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0, 
                 a_dim=None, z_dim=None, y_dim=1):
        super().__init__()
        self.lambda1 = lambda1  # redundancy penalty
        self.lambda2 = lambda2  # info maximization
        self.lambda3 = lambda3  # smoothness
        self.lambda4 = lambda4  # recon
        assert a_dim is not None and z_dim is not None
        self.critic = DonskerCritic(a_dim, z_dim)
        self.scorer = BilinearScorer(a_dim, y_dim)

    def forward(self, Z, A, Y, A_prev=None, recon=None, target=None):
        """
        Z: [B, N, d] (node embeddings, batch)
        A: [B, N, N] (adjacency matrices)
        Y: [B, ...]   (labels or targets for InfoNCE)
        A_prev: [B, N, N] (adj matrix previous timestep)
        recon: [B, N, d]  (reconstructed features)
        target: [B, N, d] (ground-truth features)
        """
        B, N, d = Z.shape

        # === 1. Self-expression loss: Z ≈ A @ Z ===
        Z_hat = torch.bmm(A, Z)          # [B, N, d]
        L_rec = F.mse_loss(Z_hat, Z)

        # === 2. Redundancy loss: Donsker-Varadhan upper bound on I(A;Z) ===
        # Flatten A for each batch
        A_flat = A.view(B, -1)           # [B, N*N]
        Z_flat = Z.mean(dim=1)           # [B, d] (global mean pooling)
        # Positive (joint) samples
        T_pos = self.critic(A_flat, Z_flat).mean()
        # Negative samples: shuffle batch
        idx_perm = torch.randperm(B)
        T_neg = torch.exp(self.critic(A_flat[idx_perm], Z_flat)).mean()
        L_redundancy = T_pos - torch.log(T_neg + 1e-8)

        # === 3. Info term: InfoNCE lower bound on I(A;Y) ===
        # Assume Y is [B, y_dim], can expand if needed
        pos_score = self.scorer(A_flat, Y).squeeze(-1)
        Y_neg = Y[torch.randperm(B)]
        neg_score = self.scorer(A_flat, Y_neg).squeeze(-1)
        logits = torch.stack([pos_score, neg_score], dim=1)   # [B, 2]
        labels = torch.zeros(B, dtype=torch.long, device=Z.device)
        L_info = F.cross_entropy(logits, labels)

        # === 4. Temporal smoothness: L2 between A and A_prev ===
        L_smooth = torch.tensor(0.0, device=Z.device)
        if A_prev is not None:
            L_smooth = F.mse_loss(A, A_prev)

        # === 5. Reconstruction loss: only for masked nodes ===
        L_recon = torch.tensor(0.0, device=Z.device)
        if recon is not None and target is not None:
            L_recon = F.mse_loss(recon, target)

        # === 6. Total IB-Graph loss and final loss ===
        L_ib = L_rec + self.lambda1 * L_redundancy - self.lambda2 * L_info
        L_total = L_ib + self.lambda3 * L_smooth + self.lambda4 * L_recon

        return L_total, {
            "ib": L_ib.item(),
            "rec": L_rec.item(),
            "redundancy": L_redundancy.item(),
            "info": L_info.item(),
            "smooth": L_smooth.item(),
            "recon": L_recon.item(),
            "total": L_total.item()
        }
