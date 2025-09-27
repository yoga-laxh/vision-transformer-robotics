# datasets/mvtec.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class MVTecDataset(Dataset):
    def __init__(self, root, category="bottle", split="train", transform=None):
        self.root = root
        self.category = category
        self.split = split  # train/test
        self.transform = transform if transform else T.ToTensor()

        self.data_dir = os.path.join(root, category, split)
        self.samples = []

        for defect_type in os.listdir(self.data_dir):
            defect_dir = os.path.join(self.data_dir, defect_type)
            for fname in os.listdir(defect_dir):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(defect_dir, fname), defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
