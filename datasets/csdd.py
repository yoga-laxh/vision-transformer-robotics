# datasets/csdd.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CSDDDataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform if transform else T.ToTensor()

        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)

        self.samples = []
        for fname in os.listdir(self.img_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                lbl_file = fname.replace(".jpg", ".txt").replace(".png", ".txt")
                self.samples.append((os.path.join(self.img_dir, fname),
                                     os.path.join(self.lbl_dir, lbl_file)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        labels = open(lbl_path).read().strip()

        if self.transform:
            image = self.transform(image)

        return image, labels
