import torch
import torchvision.transforms as T
from PIL import Image

class BusinessCardAugmentation:
    """
    Applies real-world augmentations to simulate camera capture artifacts:
    blur, rotation, color jitter, perspective distortion.
    """
    def __init__(self, img_size=(1024, 1024)):
        self.img_size = img_size
        
        self.train_transforms = T.Compose([
            T.Resize(self.img_size),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.RandomRotation(degrees=(-15, 15)),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transforms = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, img, split='train'):
        if split == 'train':
            return self.train_transforms(img)
        return self.val_transforms(img)

if __name__ == "__main__":
    dummy_img = Image.new('RGB', (1050, 600), color=(255, 255, 255))
    aug = BusinessCardAugmentation()
    tensor_img = aug(dummy_img, split='train')
    print("Augmented tensor shape:", tensor_img.shape)
