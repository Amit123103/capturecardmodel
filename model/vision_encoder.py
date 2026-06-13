import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and linearly embeds them.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # x: (B, C, H, W) -> (B, embed_dim, grid_H, grid_W) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class MultiScaleFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features to capture both fine details (text) and layout structure.
    """
    def __init__(self, embed_dims=[128, 256, 512, 1024]):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dims[3]),
            nn.GELU()
        )
        
    def forward(self, x):
        features = []
        x = self.stage1(x)
        features.append(x)
        x = self.stage2(x)
        features.append(x)
        x = self.stage3(x)
        features.append(x)
        x = self.stage4(x)
        features.append(x)
        return features

class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(),
            nn.Conv2d(dim // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attn = self.attention(x)
        return x * attn

class VisionEncoder(nn.Module):
    """
    Complete Vision Encoder for the Business Card Model.
    Combines hierarchical feature extraction with spatial attention for layout understanding.
    """
    def __init__(self, img_size=1024, in_chans=3, embed_dim=768, num_heads=12, depth=6):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=16, in_chans=in_chans, embed_dim=embed_dim)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.multi_scale_extractor = MultiScaleFeatureExtractor()
        
        self.spatial_attn = SpatialAttention(dim=1024)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        try:
            seq_x = self.patch_embed(x)
            seq_x = seq_x + self.pos_embed
            sequence_features = self.transformer_encoder(seq_x)
        except Exception:
            sequence_features = None 
            
        ms_features = self.multi_scale_extractor(x)
        layout_features = self.spatial_attn(ms_features[-1])
        
        return {
            "sequence_features": sequence_features,
            "multi_scale_features": ms_features,
            "layout_features": layout_features
        }

if __name__ == "__main__":
    x = torch.randn(2, 3, 1024, 1024)
    model = VisionEncoder(img_size=1024)
    out = model(x)
    print("Sequence features shape:", out["sequence_features"].shape)
    print("Layout features shape:", out["layout_features"].shape)
