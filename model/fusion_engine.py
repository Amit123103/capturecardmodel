import torch
import torch.nn as nn

class MultimodalFusionEngine(nn.Module):
    """
    Fuses visual features from the Vision Encoder with text/semantic context.
    Uses Cross-Attention where text embeddings query visual features to extract layout/value information.
    """
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, text_embeds, visual_embeds):
        """
        text_embeds: (B, SeqLen, D) - Embeddings of the current text/schema context (e.g., "Company Name: ")
        visual_embeds: (B, N, D) - Patch sequence features from Vision Encoder
        """
        x = text_embeds
        for attn, norm1, norm2, ffn in zip(self.layers, self.norms1, self.norms2, self.ffns):
            attn_out, _ = attn(query=x, key=visual_embeds, value=visual_embeds)
            x = norm1(x + attn_out)
            
            ffn_out = ffn(x)
            x = norm2(x + ffn_out)
            
        return x
