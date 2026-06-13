import os
import zipfile

def extract_dataset():
    zip_path = r"C:\Users\amita\myprojects\cardcapturemodel\archive (1).zip"
    extract_dir = r"C:\Users\amita\myprojects\cardcapturemodel\custom_data"
    
    if not os.path.exists(zip_path):
        print(f"Zip file not found at {zip_path}")
        return
        
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    print("Extraction complete.")
    if os.path.exists(extract_dir):
        print("Contents:", os.listdir(extract_dir))

if __name__ == "__main__":
    extract_dataset()
