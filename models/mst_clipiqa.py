import torch
import torch.nn as nn
from .mstfe import MSTFE
from .gff import GFF
from .heads import TemplateHead, PromptHead

class MSTCLIPIQA(nn.Module):
    def __init__(self, variant='A'):
        """
        MST-CLIPIQA Model Wrapper.
        
        Args:
            variant (str): 'A' for Template-based Head (No Prompts), 
                           'B' for Prompt-Anchored Head (With Prompts).
        """
        super(MSTCLIPIQA, self).__init__()
        self.variant = variant
        
        # 1. Multi-Scale Two-Stream Feature Extraction
        self.mstfe = MSTFE()
        
        # 2. Gated Feature Fusion
        self.gff = GFF()
        
        # 3. Prediction Head
        if self.variant == 'A':
            self.head = TemplateHead()
        elif self.variant == 'B':
            self.head = PromptHead()
        else:
            raise ValueError("Variant must be 'A' or 'B'")
            
    def forward(self, x, prompts=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            prompts (list of str, optional): Text prompts for Variant B.
            
        Returns:
            torch.Tensor: Predicted quality scores of shape (B, 1).
        """
        # Extract coarse and fine features
        f_c, f_f = self.mstfe(x)
        
        # Fuse features
        z = self.gff(f_c, f_f)
        
        # Predict score
        if self.variant == 'A':
            q_hat = self.head(z)
        else:
            if prompts is None:
                raise ValueError("Prompts must be provided for Variant B")
            q_hat = self.head(z, prompts)
            
        return q_hat
