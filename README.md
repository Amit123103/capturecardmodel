# Business Card Foundation Model

This repository contains the from-scratch architecture for a multimodal foundation model designed to extract information from business cards.

## Prerequisites

Before starting, ensure you have Python 3.8+ installed along with the required libraries.

```bash
pip install torch torchvision pillow
```

## How to Train Using a ZIP File Dataset

If you have downloaded a custom dataset as a `.zip` file, follow these steps to train the model.

### 1. Expected Dataset Structure

Your `.zip` file should contain the dataset formatted with an `images` folder and a `labels` folder. The labels should be in JSON format corresponding to each image.

```text
dataset.zip
│
├── images/
│   ├── card_001.png
│   ├── card_002.jpg
│   └── ...
│
└── labels/
    ├── card_001.json
    ├── card_002.json
    └── ...
```

### 2. Extract the Dataset

Extract the zip file into the project directory (or any directory on your machine). You can use Python to extract it quickly:

```python
# extract_dataset.py
import zipfile
import os

zip_path = "path/to/your/dataset.zip"
extract_dir = "custom_data"

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print(f"Extracted dataset to {extract_dir}/")
```

### 3. Update the Training Script

Open the `tests/test_architecture.py` (or create a new `train.py` script based on `train_pipeline.py`) and point the `BusinessCardDataset` to your extracted folder.

Example `train.py`:
```python
import torch
from torch.utils.data import DataLoader
from training.train_pipeline import FoundationModelPipeline
from training.dataset import BusinessCardDataset

def train_model():
    print("Initializing Pipeline...")
    # Change device to 'cuda' if you have an NVIDIA GPU
    pipeline = FoundationModelPipeline(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Point the dataset to your extracted zip folder
    data_dir = "custom_data" # <-- The folder you extracted the zip into
    
    print(f"Loading dataset from {data_dir}...")
    dataset = BusinessCardDataset(data_dir=data_dir, tokenizer=pipeline.tokenizer, split='train')
    
    # Optional: Build vocabulary from your JSON files if not using a pre-built one
    # ...
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("Starting Training Loop...")
    epochs = 10
    for epoch in range(epochs):
        loss = pipeline.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

if __name__ == "__main__":
    train_model()
```

### 4. Run the Training

Execute your training script from the terminal:

```bash
python train.py
```

## Running the Tests

To verify that the neural network architecture shapes match up correctly without needing a real dataset, you can run the built-in synthetic test:

```bash
python tests/test_architecture.py
```
