'''
Sample predictive model.
The ingestion program will call `predict` to get a prediction for each test image and then save the predictions for scoring. The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
'''
from open_clip import create_model
from torchvision import transforms
import torch
import pickle
import os

class Model:
    def __init__(self):
        # model will be called from the load() method
        self.clf = None

    def load(self):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

        model = create_model("hf-hub:imageomics/bioclip", output_dict=True, require_pretrained=True)
        self.model = model.to(self.device)
                
        with open(os.path.join(os.path.dirname(__file__), "clf.pkl"), "rb") as f:
            self.clf = pickle.load(f)

        self.preprocess_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            ]
        )
    

    def predict(self, datapoint):
        
        with torch.no_grad():
            image = self.preprocess_img(datapoint).to(self.device)
            image_feature = self.model(image.unsqueeze(0))['image_features']
            image_feature = image_feature.detach().cpu().numpy()
            score = self.clf.predict_proba(image_feature)[:, 1][0]
        
        return score
