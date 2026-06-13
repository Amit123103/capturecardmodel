import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    """
    Autoregressive Decoder for generating structured JSON output or plain text.
    """
    def __init__(self, vocab_size, embed_dim=768, num_heads=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(2048, embed_dim) 
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, tgt, memory):
        """
        tgt: (B, tgt_seq_len) - previous tokens
        memory: (B, mem_seq_len, embed_dim) - fused multimodal features
        """
        B, seq_len = tgt.shape
        
        positions = torch.arange(0, seq_len, device=tgt.device).unsqueeze(0).expand(B, seq_len)
        tgt_emb = self.embedding(tgt) + self.pos_embed(positions)
        
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        logits = self.fc_out(output)
        return logits
