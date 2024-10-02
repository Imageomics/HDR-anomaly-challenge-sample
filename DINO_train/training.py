import csv
from pathlib import Path

import torch
import joblib 
from torch.utils.data import DataLoader

from dataset import ButterflyDataset
from data_utils import data_transforms, load_data
from evaluation import evaluate, print_evaluation, print_major_minor_stats
from model_utils import get_feats_and_meta, get_dino_model
from classifier import train, get_scores

# Configuration         
ROOT_DATA_DIR = Path("/path/to/your/project/data")
DATA_FILE = ROOT_DATA_DIR / "ref" / "butterfly_anomaly_train.csv"
IMG_DIR = ROOT_DATA_DIR / "images"
CLF_SAVE_DIR = Path("/path/to/your/project/models/trained_clfs")
DEVICE = "cuda:1"
BATCH_SIZE = 4

def setup_data_and_model():
    # Load Data
    train_data, test_data = load_data(DATA_FILE, IMG_DIR)
    global MAJOR_CAMS, MINOR_CAMS
    MAJOR_CAMS = test_data[test_data["ssp_indicator"] == "major"]["CAMID"].tolist()
    MINOR_CAMS = test_data[test_data["ssp_indicator"] == "minor"]["CAMID"].tolist()

    # Model setup
    model = get_dino_model()
    return model.to(DEVICE), train_data, test_data

def prepare_data_loaders(train_data, test_data):
    train_sig_dset = ButterflyDataset(train_data, IMG_DIR, transforms=data_transforms())
    tr_sig_dloader = DataLoader(train_sig_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_dset = ButterflyDataset(test_data, IMG_DIR, transforms=data_transforms())
    test_dl = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    return tr_sig_dloader, test_dl

def extract_features(tr_sig_dloader, test_dl, model):
    tr_features, tr_labels, tr_camids = get_feats_and_meta(tr_sig_dloader, model, DEVICE)
    test_features, test_labels, test_camids = get_feats_and_meta(test_dl, model, DEVICE)
    return tr_features, tr_labels, test_features, test_labels, test_camids

def train_and_evaluate(tr_features, tr_labels, test_features, test_labels, test_camids):
    configs = ["svm", "sgd", "knn"]
    csv_output = []

    for con in configs:
        print(f"Training and evaluating {con}...")
        clf, acc, h_acc, nh_acc = train(tr_features, tr_labels, con)

        # Save model to the specified path
        model_filename = CLF_SAVE_DIR / f"trained_{con}_classifier.pkl"
        joblib.dump(clf, model_filename)
        print(f"Saved {con} classifier to {model_filename}")
        print(f"{con}: Acc - {acc:.4f}, Hacc - {h_acc:.4f}, NHacc - {nh_acc:.4f}")
        scores = get_scores(clf, test_features)
        eval_scores = evaluate(scores, test_labels, reversed=False)
        print_major_minor_stats(scores, test_labels, test_camids, MAJOR_CAMS, MINOR_CAMS, reversed=False)        
        print_evaluation(*eval_scores)
        csv_output.append([f"BioCLIP Features + {con}"] + list(eval_scores))
    return csv_output


def main():
    model, train_data, test_data = setup_data_and_model()
    tr_sig_dloader, test_dl = prepare_data_loaders(train_data, test_data)
    tr_features, tr_labels, test_features, test_labels, test_camids = extract_features(tr_sig_dloader, test_dl, model)
    csv_output = train_and_evaluate(tr_features, tr_labels, test_features, test_labels, test_camids)
    
if __name__ == "__main__":
    main()