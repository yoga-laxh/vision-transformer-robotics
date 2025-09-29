# datasets/msdd.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class MSDDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform if transform else T.ToTensor()

        self.data_dir = os.path.join(root, split)
        self.samples = []
        classes = os.listdir(self.data_dir)

        for cls in classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(".jpg") or fname.endswith(".png"):
                    self.samples.append((os.path.join(cls_dir, fname), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
