# models/simclr.py
import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, base_encoder="resnet50", projection_dim=128):
        super(SimCLR, self).__init__()

        # 1️⃣ Encoder backbone (e.g., ResNet-18 / ResNet-50)
        if base_encoder == "resnet18":
            self.encoder = models.resnet18(weights=None)
        elif base_encoder == "resnet50":
            self.encoder = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported encoder: {base_encoder}")

        # Remove final classification layer (fc)
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # 2️⃣ Projection head (2-layer MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, projection_dim)
        )

    def forward(self, x_i, x_j):
        """
        Forward two augmented images through encoder + projection head.
        Returns projected feature vectors (z_i, z_j).
        """
        h_i = self.encoder(x_i)  # features before projection
        h_j = self.encoder(x_j)

        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)

        # Normalize to unit hypersphere (important for contrastive loss)
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        return z_i, z_j
