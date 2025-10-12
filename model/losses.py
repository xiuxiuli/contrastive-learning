# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: normalized projection vectors from SimCLR model.
        shape = (batch_size, projection_dim)
        """
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Cosine similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2N, 2N)
        # Remove self-similarity (diagonal elements)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: (i, i+N) and (i+N, i)
        positives = torch.cat([torch.arange(batch_size, 2 * batch_size),
                               torch.arange(0, batch_size)]).to(z.device)
        positives = torch.stack([torch.arange(2 * batch_size).to(z.device), positives], dim=1)

        # Extract similarities of positive pairs
        sim_pos = torch.exp(sim_matrix[positives[:, 0], positives[:, 1]])

        # Denominator: sum over all except self
        sim_all = torch.exp(sim_matrix).sum(dim=1)

        loss = -torch.log(sim_pos / sim_all)
        loss = loss.mean()

        return loss
