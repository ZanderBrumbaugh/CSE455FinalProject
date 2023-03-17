import torch
import torch.nn as nn

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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

class DepthDataset(Dataset):
    def __init__(self, data_dir, dual_transform):
        super(DepthDataset, self).__init__()
        self.paths = parseDataDir(data_dir)
        self.dual_transform = dual_transform

    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        img_path, depth_path = self.paths[idx]
        
        depth_npy = np.load(depth_path)
        assert depth_npy.dtype == np.float32
        
        depth = torch.from_numpy(depth_npy)#.astype(np.int32))
        
        # ensure depth is of shape (1, H, W)
        depth = depth.squeeze().unsqueeze(0)

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
#             img = self.transform(img)

        return self.dual_transform(img, depth)

