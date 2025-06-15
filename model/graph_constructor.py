import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConstructor(nn.Module):
    """
    信息瓶颈引导的动态图构造器，严格遵循IB-Graph methodology。
    """
    def __init__(self, num_nodes, embedding_dim, top_k=8, hidden_dim=128):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.top_k = top_k

        # 输入相关的动态邻接生成器（自表达门控 MLP）
        self.graph_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes)
        )

    def forward(self, Z_t, Y_t=None, Z_prev=None):
        """
        Z_t: [N, d] 当前时间节点嵌入
        Y_t: [N, ...] 当前标签（用于互信息估计，可选）
        Z_prev: [N, d] 上一时刻节点嵌入
        Returns:
            Z_hat: 自表达重建 [N, d]
            A_norm: min-max归一化邻接 [N, N]
            A_sparse: top-k稀疏邻接 [N, N]
            phi: 边权置信度 [N, N]
            losses: dict
        """

        # 动态邻接矩阵 A_t = f(Z_t): [N, N]
        A_logits = []
        for i in range(self.num_nodes):
            # 每个节点 i 用自身向量通过mlp预测与所有节点相关性
            A_i = self.graph_mlp(Z_t[i])  # [N]
            A_logits.append(A_i)
        A_t = torch.stack(A_logits, dim=0).abs()  # [N, N], abs保证正权重

        # Min-Max归一
        A_min, A_max = A_t.min(dim=-1, keepdim=True)[0], A_t.max(dim=-1, keepdim=True)[0]
        A_norm = (A_t - A_min) / (A_max - A_min + 1e-8)

        # Top-K稀疏
        A_sparse = torch.zeros_like(A_norm)
        topk = torch.topk(A_norm, self.top_k, dim=-1)
        indices = topk.indices
        row_idx = torch.arange(self.num_nodes).unsqueeze(1).expand(-1, self.top_k)
        A_sparse[row_idx, indices] = A_norm[row_idx, indices]

        # Eq (1) 自表达重建
        Z_hat = torch.matmul(A_sparse, Z_t)  # [N, d]
        L_reconstruct = F.mse_loss(Z_hat, Z_t)

        # Eq (2) 信息瓶颈正则项（接口，具体计算建议在主模型里实现/调用）
        # InfoNCE 下界用于判别增强 mutual info，DV上界用于冗余约束
        L_IB_predict = torch.tensor(0.0, device=Z_t.device)
        L_IB_redundancy = torch.tensor(0.0, device=Z_t.device)
        if (Y_t is not None):
            # 这里仅接口，建议主模型外部实现g_phi评分函数
            pass
        # 若主模型实现了T_psi, 也可传入，留接口

        # Eq (3) 时间一致性正则（A_t vs 上一时刻A_{t-1}）
        L_smooth = torch.tensor(0.0, device=Z_t.device)
        if Z_prev is not None:
            with torch.no_grad():
                A_prev_logits = []
                for i in range(self.num_nodes):
                    A_i = self.graph_mlp(Z_prev[i])
                    A_prev_logits.append(A_i)
                A_prev = torch.stack(A_prev_logits, dim=0).abs()
                A_prev_min, A_prev_max = A_prev.min(dim=-1, keepdim=True)[0], A_prev.max(dim=-1, keepdim=True)[0]
                A_prev_norm = (A_prev - A_prev_min) / (A_prev_max - A_prev_min + 1e-8)
            L_smooth = F.mse_loss(A_norm, A_prev_norm)

        phi = A_sparse.abs()

        return Z_hat, A_norm, A_sparse, phi, {
            "reconstruct": L_reconstruct,
            "smooth": L_smooth,
            "IB_predict": L_IB_predict,
            "IB_redundancy": L_IB_redundancy
        }


if __name__ == '__main__':
    N, d = 32, 64
    Z = torch.randn(N, d)
    model = GraphConstructor(num_nodes=N, embedding_dim=d, top_k=5)
    Z_hat, A, A_sparse, phi, losses = model(Z)
    print('Z_hat shape:', Z_hat.shape)
    print('A shape:', A.shape)
    print('Reconstruct loss:', losses['reconstruct'].item())
