import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    def __init__(self, lambda_rank=1.0, margin=0.1):
        super(CompositeLoss, self).__init__()
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.mse_loss = nn.MSELoss()

    def forward(self, q_hat, q):
        """
        Args:
            q_hat: Predicted quality scores, shape (B, 1) or (B,)
            q: Ground truth MOS, shape (B, 1) or (B,)
        Returns:
            Total composite loss
        """
        q_hat = q_hat.view(-1)
        q = q.view(-1)
        
        # 1. Mean Squared Error Loss
        l_mse = self.mse_loss(q_hat, q)
        
        # 2. Pairwise Margin Ranking Loss
        # Create pairwise differences
        q_diff = q.unsqueeze(0) - q.unsqueeze(1) # shape (B, B)
        q_hat_diff = q_hat.unsqueeze(0) - q_hat.unsqueeze(1) # shape (B, B)
        
        # target_sign = sign(q_i - q_j)
        target_sign = torch.sign(q_diff)
        
        # rank_loss = mean(ReLU(-target_sign * (q_hat_i - q_hat_j) + margin))
        # We only consider pairs where i != j to avoid self-comparison, 
        # but since target_sign is 0 when i=j, ReLU(0 + margin) would add margin.
        # So we should mask out the diagonal or only take upper/lower triangle.
        # Actually, the formula is usually applied to all pairs or valid pairs.
        # Let's apply it to all pairs where target_sign != 0.
        
        mask = (target_sign != 0).float()
        if mask.sum() > 0:
            rank_loss_matrix = F.relu(-target_sign * q_hat_diff + self.margin)
            l_rank = (rank_loss_matrix * mask).sum() / mask.sum()
        else:
            l_rank = torch.tensor(0.0, device=q_hat.device)
            
        # Total Loss
        loss = l_mse + self.lambda_rank * l_rank
        
        return loss
