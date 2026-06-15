import os
import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import easyocr

from training.train_pipeline import FoundationModelPipeline
from data.augmentation import BusinessCardAugmentation

# Initialize the Application
app = FastAPI(title="Business Card Extractor API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading Models on {device}...")

# 1. Load the Foundation Model Pipeline
pipeline = FoundationModelPipeline(device=device)
pipeline.tokenizer.build_vocab(["dummy initialization vocab for inference test"])
augmentor = BusinessCardAugmentation()

# 2. Load the Production OCR Fallback (EasyOCR) to guarantee plain text details
print("Loading EasyOCR text extraction engine...")
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

os.makedirs("static", exist_ok=True)

import re

def parse_business_card(ocr_lines):
    details = {
        "Company Name": "",
        "Name": "",
        "Mobile Number": "",
        "Email": "",
        "Address": "",
        "Website": "",
        "Other Details": []
    }
    
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    phone_pattern = re.compile(r'(\+?\d{1,3}[-.\s]??\(?\d{1,4}\)?[-.\s]??\d{1,4}[-.\s]??\d{1,4}[-.\s]??\d{1,9})')
    website_pattern = re.compile(r'(?:www\.|http:|https:|)[a-zA-Z0-9-]+\.(?:com|org|net|io|co|in|uk)\b', re.IGNORECASE)
    
    unassigned_lines = []
    
    for line in ocr_lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        # Match Email
        email_match = email_pattern.findall(line_clean)
        if email_match and not details["Email"]:
            details["Email"] = email_match[0]
            continue
            
        # Match Phone (looking for digits and common separators)
        # Check if it has at least 7 digits to avoid matching random numbers
        digits_only = re.sub(r'\D', '', line_clean)
        if len(digits_only) >= 7 and phone_pattern.search(line_clean):
            # Try to grab the longest number-like string
            if not details["Mobile Number"]:
                details["Mobile Number"] = line_clean
            else:
                details["Mobile Number"] += f" / {line_clean}"
            continue
            
        # Match Website
        website_match = website_pattern.findall(line_clean)
        if website_match and ' ' not in line_clean and not details["Website"]:
            details["Website"] = line_clean
            continue
            
        unassigned_lines.append(line_clean)
        
    # Heuristics for the remaining lines
    if unassigned_lines:
        # Usually, Company name is the first prominent text, or Name is.
        # Let's assign the first line to Company Name
        details["Company Name"] = unassigned_lines.pop(0)
        
    if unassigned_lines:
        # Assign second line to Name (or Job Title)
        details["Name"] = unassigned_lines.pop(0)
        
    # Any lines containing typical address keywords go to Address
    address_keywords = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'floor', 'fl', 'block', 'city', 'state', 'p.o.', 'box', 'suite']
    remaining = []
    
    for line in unassigned_lines:
        lower_line = line.lower()
        if any(kw in lower_line for kw in address_keywords) or (len(line) > 15 and any(char.isdigit() for char in line)):
            if not details["Address"]:
                details["Address"] = line
            else:
                details["Address"] += f", {line}"
        else:
            remaining.append(line)
            
    details["Other Details"] = remaining
    
    # Clean up empty fields to not show them
    final_details = {k: v for k, v in details.items() if v}
    return final_details

def self_learn_task(img_tensor, extracted_text):
    """
    Background task that trains the foundation model using the OCR text as ground truth.
    """
    try:
        print("\n[Self-Learning] Starting background training step...")
        # Encode the text into tokens
        tokens = pipeline.tokenizer.encode(extracted_text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0) # Batch size 1
        
        # Run a single training step
        pipeline.vision_encoder.train()
        pipeline.fusion_engine.train()
        pipeline.decoder.train()
        
        loss = pipeline.train_step(img_tensor, tokens_tensor)
        print(f"[Self-Learning] Training step completed. Loss: {loss:.4f}\n")
    except Exception as e:
        print(f"[Self-Learning] Error during background training: {e}")

@app.post("/extract")
async def extract_card(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image for the background training task
        img_tensor = augmentor(image, split='val').unsqueeze(0)
        
        # --- FAST EXTRACTION PATH (EasyOCR) ---
        # Resize image to a maximum dimension of 1000px to drastically speed up OCR
        ocr_image = image.copy()
        ocr_image.thumbnail((1000, 1000))
        img_np = np.array(ocr_image)
        ocr_results = reader.readtext(img_np, detail=0, paragraph=False)
        
        # Parse the structured fields
        structured_data = parse_business_card(ocr_results)
        
        # Format as perfectly structured plain text
        if structured_data:
            lines = []
            for k, v in structured_data.items():
                if isinstance(v, list):
                    if v:
                        lines.append(f"{k}:\n  - " + "\n  - ".join(v))
                else:
                    lines.append(f"{k}: {v}")
            plain_text_details = "\n".join(lines)
        else:
            plain_text_details = "No text could be found in the image. Please try a clearer picture."

        # Add the slow learning task to the background
        # This way, the user gets their fast response immediately!
        background_tasks.add_task(self_learn_task, img_tensor, plain_text_details)

        return JSONResponse(content={
            "success": True,
            "extracted_text": plain_text_details, # Giving you the perfect structured details
            "device_used": device
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # access_log=False hides the "304 Not Modified" and "200 OK" messages
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, access_log=False)
