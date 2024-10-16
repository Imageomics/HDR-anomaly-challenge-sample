import torch
import numpy as np
from tqdm import tqdm

def get_feats_and_meta(dloader, model, device, ignore_feats=False):
    all_feats = None
    labels = []

    for img, lbl in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            feats = None
            if not ignore_feats:
                out = model(img.to(device))['image_features']
                feats = out.cpu().numpy()
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats
                
        labels.extend(lbl.cpu().numpy().tolist())
        
    labels = np.array(labels)
    return all_feats, labels
