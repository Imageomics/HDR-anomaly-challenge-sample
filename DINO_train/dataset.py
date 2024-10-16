from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class ButterflyDataset(Dataset):
    def __init__(self, data, root_dir, transforms=None):
        self.data = data
        self.root_dir = root_dir
        self.transforms = transforms

        # Validate the 'hybrid_stat' column to ensure it contains only expected values
        valid_classes = {"hybrid", "non-hybrid"}
        self.data["hybrid_stat"] = self.data["hybrid_stat"].str.strip().str.lower()  # Normalize the values
        if not set(self.data["hybrid_stat"].unique()).issubset(valid_classes):
            raise ValueError("Unexpected values found in 'hybrid_stat' column.")

        # Define classes explicitly to avoid relying on sorted order
        self.classes = ["non-hybrid", "hybrid"]
        self.cls_lbl_map = {cls: i for i, cls in enumerate(self.classes)}

        # Generate labels using a vectorized approach for efficiency
        self.labels = self.data["hybrid_stat"].map(self.cls_lbl_map).tolist()

        print("Created base dataset with {} samples".format(len(self.data)))
        
    def get_file_path(self, x):
        return os.path.join(self.root_dir, x['filename'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        img_path = self.get_file_path(x)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")
        
        lbl = self.labels[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, lbl

