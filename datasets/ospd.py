# datasets/ospd.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class OSPDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split  # train/val/test
        self.transform = transform if transform else T.ToTensor()

        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        self.samples = [(f, f.replace(".jpg", ".txt")) for f in os.listdir(self.img_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, ann_file = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_file)
        ann_path = os.path.join(self.ann_dir, ann_file)

        image = Image.open(img_path).convert("RGB")
        labels = open(ann_path).read().strip()

        if self.transform:
            image = self.transform(image)

        return image, labels
