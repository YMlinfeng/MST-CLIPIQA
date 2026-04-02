import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPTextModel

class TemplateHead(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(model_name)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.temperature = nn.Parameter(torch.tensor(100.0))
        self.register_buffer("quality_scores", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        
        # Precompute text features
        templates = [f"a photo with {q} quality" for q in ['bad', 'poor', 'fair', 'good', 'perfect']]
        inputs = self.tokenizer(templates, padding=True, return_tensors="pt")
        with torch.no_grad():
            text_features = self.text_encoder(**inputs).text_embeds
            text_features = F.normalize(text_features, dim=-1)
        self.register_buffer("text_features", text_features)

    def forward(self, z, prompts=None):
        # z: (Batch, 512)
        z = F.normalize(z, dim=-1)
        # text_features: (5, 512)
        # sim: (Batch, 5)
        sim = torch.matmul(z, self.text_features.T)
        sim = sim * self.temperature
        probs = F.softmax(sim, dim=-1)
        score = torch.sum(probs * self.quality_scores, dim=-1, keepdim=True)
        return score

class PromptHead(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        self.W_1 = nn.Linear(512, 512)
        self.W_2 = nn.Linear(512, 512)
        self.w_o = nn.Linear(512, 1)

    def forward(self, z, prompts):
        # z: (Batch, 512)
        # prompts: list of strings
        device = z.device
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # E_text(T): (Batch, Seq_Len, 512)
            text_outputs = self.text_encoder(**inputs)
            text_features = text_outputs.last_hidden_state
            
        # Q: (Batch, 1, 512)
        Q = z.unsqueeze(1)
        # K, V: (Batch, Seq_Len, 512)
        K = text_features
        V = text_features
        
        attn_output, _ = self.cross_attn(Q, K, V)
        # attn_output: (Batch, 1, 512)
        
        z_prime = Q + self.alpha * attn_output
        z_prime = z_prime.squeeze(1) # (Batch, 512)
        
        q_hat = self.w_o(F.gelu(self.W_1(z_prime)) + self.W_2(z_prime))
        return q_hat
