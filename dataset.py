import os, torch
import numpy as np
from PIL import Image

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
        return len(self.imgs)
        
    def __getitem__(self, idx):
        img_path, depth_path = self.paths[idx]
        depth = np.load(depth_path)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        return img, depth