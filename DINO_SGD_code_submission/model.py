'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import os
import pickle

import torch
from torchvision import transforms
from transformers import AutoModel

import numpy as np
import PIL

from sklearn.linear_model import SGDClassifier

class Model:
    def __init__(self):
        self.dino_name = 'facebook/dinov2-base'
        self.pil_transform_fn = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        
    def load(self):
        # DINO backbone
        model = AutoModel.from_pretrained(self.dino_name)
        model.eval()
        self.model = model.to(self.device)
        
        # Classifier
        non_hybrid_weight = 1
        hybrid_weight = 1
        class_weights = {0: non_hybrid_weight, 1: hybrid_weight}
        
        # Load Classifier weights
        with open(os.path.join(os.path.dirname(__file__), "clf.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        
        
    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)[0]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py#L707
        cls_token = feats[:, 0]
        patch_tokens = feats[:, 1:]
        feats = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        return feats   
    
    def _get_clf_prediction(self, features: torch.Tensor) -> float:
        np_features = features.detach().cpu().numpy() # Convert to numpy for classifer compatibility
        return self.clf.predict_proba(np_features)[0, 1] # Since a batch of 1, just extract float

    def predict(self, x: PIL.Image) -> float:
        x_tensor = self.pil_transform_fn(x).to(self.device).unsqueeze(0)
        features = self._get_features(x_tensor)
        prediction = self._get_clf_prediction(features)
        return prediction
