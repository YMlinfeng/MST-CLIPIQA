import torch
import torch.nn as nn

class GFF(nn.Module):
    """
    Gated Feature Fusion (GFF) module.
    Adaptively interpolates between coarse and fine features dimension-by-dimension.
    """
    def __init__(self):
        super(GFF, self).__init__()
        
        # Gating network
        self.W_h = nn.Linear(1024, 128)
        self.W_g = nn.Linear(128, 512)
        
        # Feature transformations
        self.W_c = nn.Linear(512, 512)
        self.W_f = nn.Linear(512, 512)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_c, f_f):
        """
        Args:
            f_c: Coarse features from ViT-B/32, shape (Batch, 512)
            f_f: Fine features from ViT-B/16, shape (Batch, 512)
            
        Returns:
            z: Fused features, shape (Batch, 512)
        """
        # Concatenate features for the gating network
        # Concat(f_c, f_f) -> shape (Batch, 1024)
        concat_features = torch.cat([f_c, f_f], dim=-1)
        
        # Calculate gate g
        # g = Sigmoid(W_g * ReLU(W_h * Concat(f_c, f_f) + b_h) + b_g)
        h = self.relu(self.W_h(concat_features))
        g = self.sigmoid(self.W_g(h))
        
        # Transform features
        # W_c * f_c + b_c
        c_transformed = self.W_c(f_c)
        # W_f * f_f + b_f
        f_transformed = self.W_f(f_f)
        
        # Interpolate
        # z = g * (W_c * f_c + b_c) + (1 - g) * (W_f * f_f + b_f)
        z = g * c_transformed + (1 - g) * f_transformed
        
        return z
