import os
import random
import json
from PIL import Image, ImageDraw, ImageFont

class SyntheticCardGenerator:
    def __init__(self, output_dir="synthetic_data", num_cards=10):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        self.num_cards = num_cards
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        self.names = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Williams"]
        self.titles = ["CEO", "Software Engineer", "Marketing Manager", "CTO"]
        self.companies = ["Tech Innovators", "Global Solutions", "Creative Studio", "NextGen AI"]
        self.emails = ["john@example.com", "jane@example.com", "alice@example.com", "bob@example.com"]
        self.phones = ["+1-555-0198", "+44-20-7946-0958", "+91-9876543210", "+1-800-555-1234"]
        
        self.font = ImageFont.load_default()

    def generate_card(self, idx):
        width, height = 1050, 600
        
        bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        name = random.choice(self.names)
        title = random.choice(self.titles)
        company = random.choice(self.companies)
        email = random.choice(self.emails)
        phone = random.choice(self.phones)
        
        data = {
            "name": name,
            "designation": title,
            "company": company,
            "email": email,
            "phone": phone
        }
        
        layout = {
            "name": (100, 100),
            "designation": (100, 150),
            "company": (100, 50),
            "email": (100, 400),
            "phone": (100, 450)
        }
        
        bboxes = []
        for key, value in data.items():
            x, y = layout[key]
            text_width = len(value) * 6
            text_height = 10
            draw.text((x, y), value, fill=(0, 0, 0), font=self.font)
            
            bboxes.append({
                "label": key,
                "text": value,
                "box": [x, y, x + text_width, y + text_height]
            })
            
        img_path = os.path.join(self.images_dir, f"card_{idx}.png")
        img.save(img_path)
        
        label_path = os.path.join(self.labels_dir, f"card_{idx}.json")
        with open(label_path, 'w') as f:
            json.dump({
                "image_file": f"card_{idx}.png",
                "structured_data": data,
                "bounding_boxes": bboxes
            }, f, indent=2)

    def run(self):
        print(f"Generating {self.num_cards} synthetic business cards...")
        for i in range(self.num_cards):
            self.generate_card(i)
        print("Generation complete.")

if __name__ == "__main__":
    generator = SyntheticCardGenerator(num_cards=5)
    generator.run()
