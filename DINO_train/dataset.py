from torch.utils.data import Dataset
from PIL import Image
import os

class ButterflyDataset(Dataset):
    def __init__(self, data, root_dir, transforms=None):
        self.data = data
        self.root_dir = root_dir
        self.transforms = transforms
        self.classes = sorted(list({self.create_class_key(x) for _, x in data.iterrows()}))
        self.cls_lbl_map = {cls: i for i, cls in enumerate(self.classes)}
        self.labels = [self.cls_lbl_map[self.create_class_key(x)] for _, x in data.iterrows()]
        print("Created base dataset")
        
    @staticmethod
    def create_class_key(x):
        return x["hybrid_stat"]
        
    def get_file_path(self, x):
        return os.path.join(self.root_dir, x['filename'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        img = Image.open(self.get_file_path(x))
        lbl = self.labels[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, lbl, x['CAMID']

