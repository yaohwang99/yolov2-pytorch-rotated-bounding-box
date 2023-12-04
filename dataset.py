import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def collate_fn(batch):
    images, labels = zip(*batch)

    # Stack images into a single tensor
    images = torch.stack(images, dim=0)

    return images, labels
class YOLOv2Dataset(Dataset):
    def __init__(self, root, split, transform):
        self.root_dir = os.path.join(root, split)
        self.transform = transform
        self.image_list = [filename for filename in os.listdir(self.root_dir) if filename.endswith('0.jpg')]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image.to(self.device)
        txt_name = os.path.splitext(img_name)[0] + '.txt'

        with open(txt_name, 'r') as file:
            lines = file.readlines()

        labels = []
        for line in lines:
            x, y, w, h, angle, cls= map(float, line.strip().split())
            labels.append(torch.tensor([x, y, w, h, cls, angle], device=self.device))
        return image, labels
    
