import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.vision_encoder import VisionEncoder
from model.tokenizer import BusinessCardTokenizer
from model.fusion_engine import MultimodalFusionEngine
from model.decoder import TransformerDecoder
from training.dataset import BusinessCardDataset

class FoundationModelPipeline:
    def __init__(self, vocab_size=30000, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vocab_size = vocab_size
        
        self.vision_encoder = VisionEncoder().to(self.device)
        self.fusion_engine = MultimodalFusionEngine().to(self.device)
        self.decoder = TransformerDecoder(vocab_size=vocab_size).to(self.device)
        
        self.tokenizer = BusinessCardTokenizer(mode='char')
        
        params = list(self.vision_encoder.parameters()) + \
                 list(self.fusion_engine.parameters()) + \
                 list(self.decoder.parameters())
                 
        self.optimizer = optim.AdamW(params, lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 
        
    def train_step(self, images, target_tokens):
        self.optimizer.zero_grad()
        
        images = images.to(self.device)
        target_tokens = target_tokens.to(self.device)
        
        vision_out = self.vision_encoder(images)
        visual_embeds = vision_out["sequence_features"]
        
        tgt_input = target_tokens[:, :-1]
        tgt_expected = target_tokens[:, 1:]
        
        B = images.size(0)
        semantic_embeds = torch.zeros(B, 1, 768).to(self.device) 
        
        memory = self.fusion_engine(semantic_embeds, visual_embeds)
        
        logits = self.decoder(tgt_input, memory)
        
        logits = logits.view(-1, self.vocab_size)
        tgt_expected = tgt_expected.reshape(-1)
        
        loss = self.criterion(logits, tgt_expected)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def train_epoch(self, dataloader):
        self.vision_encoder.train()
        self.fusion_engine.train()
        self.decoder.train()
        
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch['image'], batch['target_tokens'])
            total_loss += loss
            
        return total_loss / len(dataloader)

    @torch.no_grad()
    def inference(self, image_tensor, max_len=200):
        self.vision_encoder.eval()
        self.fusion_engine.eval()
        self.decoder.eval()
        
        image_tensor = image_tensor.to(self.device)
        B = image_tensor.size(0)
        
        # 1. Vision Features
        vision_out = self.vision_encoder(image_tensor)
        visual_embeds = vision_out["sequence_features"]
        if visual_embeds is None:
            return "Failed to extract visual features."
            
        # 2. Multimodal Fusion
        semantic_embeds = torch.zeros(B, 1, 768).to(self.device)
        memory = self.fusion_engine(semantic_embeds, visual_embeds)
        
        # 3. Autoregressive Decoding
        # Start with <BOS>
        bos_token = self.tokenizer.vocab.get(self.tokenizer.bos_token, 2)
        eos_token = self.tokenizer.vocab.get(self.tokenizer.eos_token, 3)
        
        generated_tokens = [bos_token]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=self.device)
            
            logits = self.decoder(tgt_tensor, memory)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            generated_tokens.append(next_token)
            
            if next_token == eos_token:
                break
                
        decoded_text = self.tokenizer.decode(generated_tokens)
        return decoded_text


if __name__ == "__main__":
    print("Initialized Training Pipeline.")
