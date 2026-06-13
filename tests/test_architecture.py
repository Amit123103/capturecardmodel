import sys
import os
import torch

# Add root directory to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_pipeline import FoundationModelPipeline
from data.synthetic_generator import SyntheticCardGenerator

def test_pipeline():
    print("Testing Foundation Model Architecture pipeline...")
    
    # 1. Generate dummy data
    generator = SyntheticCardGenerator(output_dir="synthetic_data", num_cards=2)
    generator.run()
    
    # 2. Initialize pipeline
    pipeline = FoundationModelPipeline(device='cpu')
    pipeline.tokenizer.build_vocab(["Dummy vocab text for initialization: {}", "test name email phone company"])
    
    # 3. Create a dummy batch manually
    B = 2
    images = torch.randn(B, 3, 1024, 1024)
    target_tokens = torch.randint(0, len(pipeline.tokenizer.vocab), (B, 100))
    
    print(f"Running forward pass with batch size {B}...")
    try:
        loss = pipeline.train_step(images, target_tokens)
        print(f"Success! Backward pass complete. Dummy loss: {loss:.4f}")
    except Exception as e:
        print(f"Architecture test failed: {e}")

if __name__ == "__main__":
    test_pipeline()
