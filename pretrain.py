import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.graph_constructor import GraphConstructor
from model.loss import IRENE_Loss
from model.GraphMAE import GraphMAE
from data.dataloader_detection import load_dataset_detection


def pretrain_epoch(model, graph_constructor, ib_loss_fn, loader, optimizer, device):
    model.train()
    total_loss, total_graph_loss, total_recon_loss = 0, 0, 0

    for batch in loader:
        x, y = batch['input'].to(device), batch['label'].to(device)  # x: [B, N, D]
        B, N, D = x.size()

        all_Z_hat, all_A_t, all_phi, graph_losses, recon_losses = [], [], [], [], []

        for b in range(B):
            Z_t = x[b]  # [N, D]
            Z_hat, A_t, A_sparse, phi, graph_loss_dict = graph_constructor(Z_t)
            all_Z_hat.append(Z_hat)
            all_A_t.append(A_t)
            all_phi.append(phi)
            graph_losses.append(graph_loss_dict["reconstruct"] + graph_loss_dict["smooth"])

        Z_hat_batch = torch.stack(all_Z_hat)
        A_batch = torch.stack(all_A_t)
        phi_batch = torch.stack(all_phi)
        graph_loss_val = torch.stack(graph_losses).mean()

        pred, recon_loss = model(x, A_batch, phi_batch)

        total = recon_loss + graph_loss_val
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_graph_loss += graph_loss_val.item()
        total_recon_loss += recon_loss.item()

    return total_loss, total_graph_loss, total_recon_loss


def run_pretraining():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, _, _ = load_dataset_detection(  # 使用检测数据集
        input_dir='data', raw_data_dir='data',
        train_batch_size=8, test_batch_size=8,
        time_step_size=1, max_seq_len=1,
        num_workers=2, augmentation=False,
        adj_mat_dir='', graph_type='', top_k=8,
        filter_type='', use_fft=True,
        sampling_ratio=1, seed=123, preproc_dir='data')

    graph_constructor = GraphConstructor(num_nodes=18, embedding_dim=64, top_k=8).to(device)
    ib_loss = IRENE_Loss().to(device)
    graph_mae = GraphMAE(input_dim=256, embed_dim=64).to(device)  # 假设每节点256维输入

    optimizer = optim.Adam(list(graph_constructor.parameters()) + list(graph_mae.parameters()), lr=1e-3)

    for epoch in range(1, 101):
        loss, gl, rl = pretrain_epoch(graph_mae, graph_constructor, ib_loss, train_loader, optimizer, device)
        print(f"[Epoch {epoch}] total: {loss:.4f}, graph: {gl:.4f}, recon: {rl:.4f}")


if __name__ == '__main__':
    run_pretraining()
