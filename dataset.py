import os, torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

def parseDataDir(root_dir):
  paths = []

  for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        path = os.path.join(subdir, file)
        
        if path.endswith(".png"):
            depth_path = os.path.join(subdir, file[:-4] + "_depth.npy")

            if os.path.exists(depth_path):
                paths.append((path, depth_path))

  return paths

class DepthDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(DepthDataset, self).__init__()
        self.paths = parseDataDir(data_dir)

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_path, depth_path = self.paths[idx]
        depth = torch.from_numpy(np.load(depth_path).astype(np.int32))

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            transform = transforms.Compose([transforms.PILToTensor()])
            img = transform(img)

        return img, depth
