import torch
import torch.nn as nn
import torch.nn.functional as F
from GSA_Encoder import GSAEncoder
from graph_constructor import GraphConstructor
from GraphMAE import GraphMAE

class IRENEModel_classification(nn.Module):
    def __init__(self, args, num_classes, device):
        super().__init__()
        # Graph structure-aware encoder (for downstream classification)
        self.encoder = GSAEncoder(input_dim=args.input_dim, embed_dim=args.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(args.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        #Dynamic graph constructor
        self.graph_constructor = GraphConstructor(num_nodes=args.num_nodes, embedding_dim=args.embed_dim)
        self.device = device

    def forward(self, x, seq_lengths=None, adj=None):
        # [B, N, T, D] Input → first obtain node embeddings
        z = self.encoder(x, adj)# Supports adj, ensures dynamic graph
        out = self.mlp(z.mean(dim=1))
        return out, z

    def get_node_embeddings(self, x):
        return self.encoder(x)

# =========================
# Two-stage training pipeline implementation
# =========================
def pretrain_graphmae(graphmae, pretrain_loader, optimizer, device, epochs=100):
    graphmae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in pretrain_loader:
            x = batch['x'].to(device)
            # node masking self-supervised reconstruction
            loss = graphmae(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[GraphMAE Pretrain] Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(pretrain_loader):.4f}")
    #save Encoder's weight
    torch.save(graphmae.encoder.state_dict(), "graphmae_encoder.pth")

def finetune_irenemodel(model, train_loader, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            # Use dynamic graph structure
            z = model.get_node_embeddings(x)
            _, _, adj, phi, _ = model.graph_constructor(z)
            logits, _ = model(x, adj=adj)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Classification Finetune] Epoch {epoch+1}/{epochs}  Loss: {total_loss/len(train_loader):.4f}")

def train_pipeline(args, pretrain_loader, train_loader, num_classes, device):
    # GraphMAE self-supervised pretraining
    print("==== GraphMAE Pretraining Phase ====")
    graphmae = GraphMAE(input_dim=args.input_dim, embed_dim=args.embed_dim, mask_ratio=0.1).to(device)
    optimizer1 = torch.optim.Adam(graphmae.parameters(), lr=1e-3)
    pretrain_graphmae(graphmae, pretrain_loader, optimizer1, device, epochs=args.pretrain_epochs)

    #Downstream GSAEncoder initialized with GraphMAE pretrained weights
    print("==== IRENE Model Finetune Phase ====")
    model = IRENEModel_classification(args, num_classes, device).to(device)
    encoder_weight = torch.load("graphmae_encoder.pth", map_location=device)
    model.encoder.load_state_dict(encoder_weight)

    optimizer2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    finetune_irenemodel(model, train_loader, optimizer2, device, epochs=args.finetune_epochs)

    return model

