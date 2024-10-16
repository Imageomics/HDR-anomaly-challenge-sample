import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel

def get_dino_model(dino_name='facebook/dinov2-base'):
    model = AutoModel.from_pretrained(dino_name)
    model.eval()  
    return model

    
def get_feats_and_meta(dloader, model, device, ignore_feats=False):
    all_feats = None
    labels = []

    for img, lbl in tqdm(dloader, desc="Extracting features"):
        with torch.no_grad():
            feats = None
            if not ignore_feats:
                feats = model(img.to(device))[0]
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L707
                cls_token = feats[:, 0]
                patch_tokens = feats[:, 1:]
                feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1).cpu().numpy()
            if all_feats is None:
                all_feats = feats
            else:
                all_feats = np.concatenate((all_feats, feats), axis=0) if feats is not None else all_feats
                
        labels.extend(lbl.cpu().numpy().tolist())
        
    labels = np.array(labels)
    return all_feats, labels
