# datasets/transforms.py
import torchvision.transforms as T

robotic_transforms = T.Compose([
    T.Resize((224,224)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomRotation(15),
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.GaussianBlur(3, sigma=(0.1, 2.0)),
    T.ToTensor(),
])
