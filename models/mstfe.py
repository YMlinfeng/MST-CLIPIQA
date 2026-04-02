import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection

class MSTFE(nn.Module):
    """
    Multi-Scale Two-Stream Feature Extraction (MSTFE)
    Extracts scale-decoupled representations using dual CLIP encoders.
    Coarse patches capture global semantics; fine patches capture local textures/artifacts.
    """
    def __init__(self):
        super(MSTFE, self).__init__()
        
        # Load Coarse Stream: CLIP ViT-B/32 (Patch size P_c = 32)
        self.encoder_coarse = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load Fine Stream: CLIP ViT-B/16 (Patch size P_f = 16)
        self.encoder_fine = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        
        # CRITICAL: Both encoders MUST be strictly frozen
        for param in self.encoder_coarse.parameters():
            param.requires_grad = False
            
        for param in self.encoder_fine.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        Returns:
            f_c: Coarse features of shape (B, 512)
            f_f: Fine features of shape (B, 512)
        """
        # Extract features using image_embeds which corresponds to the projected [CLS] token representation
        coarse_outputs = self.encoder_coarse(pixel_values=x)
        f_c = coarse_outputs.image_embeds
        
        fine_outputs = self.encoder_fine(pixel_values=x)
        f_f = fine_outputs.image_embeds
        
        return f_c, f_f
