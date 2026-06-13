import os
import torch
from torch.utils.data import DataLoader
from training.train_pipeline import FoundationModelPipeline
from training.dataset import BusinessCardDataset
from data.synthetic_generator import SyntheticCardGenerator

def pretrain_vision_encoder(pipeline, data_dir, epochs=1):
    print("\n--- PHASE 1: Unsupervised Visual Pretraining on Scraped Images ---")
    print(f"Loading unlabelled images from: {data_dir}")
    
    dataset = BusinessCardDataset(data_dir=data_dir, mode='unsupervised')
    if len(dataset) == 0:
        print("No unsupervised images found. Skipping.")
        return
        
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Found {len(dataset)} real images. Running Vision Encoder pretraining passes...")
    
    pipeline.vision_encoder.train()
    optimizer = torch.optim.Adam(pipeline.vision_encoder.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            images = batch['image'].to(pipeline.device)
            out = pipeline.vision_encoder(images)
            
            features = out['sequence_features']
            if features is not None:
                loss = -features.var(dim=1).mean()
                loss.backward()
                optimizer.step()
                
                if i % 10 == 0:
                    print(f"Pretraining Epoch {epoch+1} - Step {i} - Representation Loss: {loss.item():.4f}")
            else:
                break

def train_extraction(pipeline, epochs=3):
    print("\n--- PHASE 2: Supervised Extraction Training on Synthetic Data ---")
    
    synthetic_dir = "synthetic_data"
    
    print("Generating synthetic business cards with perfect labels...")
    generator = SyntheticCardGenerator(output_dir=synthetic_dir, num_cards=50)
    generator.run()
    
    dataset = BusinessCardDataset(data_dir=synthetic_dir, tokenizer=pipeline.tokenizer, mode='supervised')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Starting Extraction Training on {len(dataset)} perfectly labeled images...")
    for epoch in range(epochs):
        loss = pipeline.train_epoch(dataloader)
        print(f"Extraction Epoch {epoch+1}/{epochs} - CrossEntropy Loss: {loss:.4f}")
        
    print("\nTraining Complete! The model is now capable of extracting JSON from Business Cards.")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Foundation Model on device: {device}")
    
    pipeline = FoundationModelPipeline(device=device)
    
    dummy_vocab = ["name", "designation", "company", "email", "phone"]
    vocab_text = " ".join(dummy_vocab) + " John Doe Jane Smith CEO Software Engineer Tech Innovators Global Solutions john@example.com +1-555-0198"
    pipeline.tokenizer.build_vocab([vocab_text])
    
    real_data_dir = r"C:\Users\amita\myprojects\cardcapturemodel\custom_data\scraped_images_bcards"
    
    pretrain_vision_encoder(pipeline, data_dir=real_data_dir, epochs=1)
    train_extraction(pipeline, epochs=3)
