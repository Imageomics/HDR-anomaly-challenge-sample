import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def data_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_feats_and_meta(dloader: DataLoader, model: torch.nn.Module, device: str, ignore_feats: bool = False) -> Tuple[np.ndarray, np.ndarray, list]:
    all_feats = None
    labels = []
    camids = []

    for img, lbl, meta, _ in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            feats = None
            if not ignore_feats:
                out = model(img.to(device))['image_features']
                feats = out.detach().cpu().numpy()
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats

        labels.extend(lbl.detach().cpu().numpy().tolist())
        camids.extend(list(meta))
        
    labels = np.array(labels)
    return all_feats, labels, camids

def _filter(dataframe: pd.DataFrame, img_dir: str) -> pd.DataFrame:
    bad_row_idxs = []
    
    for idx, row in tqdm(dataframe.iterrows(), desc="Filtering bad urls"):
        fname = row['filename']
        path = os.path.join(img_dir, fname)
    
        if not os.path.exists(path):
            print(f"File not found: {path}")
            bad_row_idxs.append(idx)
        else:
            try:
                Image.open(path)
            except Exception as e:
                print(f"Error opening {path}: {e}")
                bad_row_idxs.append(idx)

    print(f"Bad rows: {len(bad_row_idxs)}")

    return dataframe.drop(bad_row_idxs)

def load_data(data_path: str, img_dir: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = _filter(pd.read_csv(data_path), img_dir)
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_data, test_data

