import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# Train classifier with improvements
def train(X, y, classifier_config="sgd"):
    # Splitting data into training and validation sets for better generalization evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    non_hybrid_weight = 1
    hybrid_weight = 1
    class_weights = {0: non_hybrid_weight, 1: hybrid_weight}

    if classifier_config == "svm":
        clf = make_pipeline(StandardScaler(), SVC(gamma='scale', C=1, class_weight='balanced', probability=True))
    elif classifier_config == "sgd":
        base_clf = SGDClassifier(
            loss="log_loss",
            alpha=0.001,
            penalty="l2",
            eta0=0.001,
            n_iter_no_change=100,
            learning_rate='adaptive',
            max_iter=1000,
            class_weight='balanced'
        )
        clf = CalibratedClassifierCV(base_clf)  # Calibrate SGD to get probability estimates
    elif classifier_config == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)  # Increased k value for better generalization
    elif classifier_config == "gaussian":
        clf = GaussianProcessClassifier(random_state=0)
    else:
        raise ValueError("Invalid classifier_config")

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    # Evaluate on the validation set
    preds = clf.predict(X_val)
    correct = preds == y_val

    hybrid_correct = correct[y_val == 1].sum()
    non_hybrid_correct = correct[y_val == 0].sum()

    acc = clf.score(X_val, y_val)
    h_acc = hybrid_correct / (y_val == 1).sum() if (y_val == 1).sum() > 0 else 0
    nh_acc = non_hybrid_correct / (y_val == 0).sum() if (y_val == 0).sum() > 0 else 0

    return clf, acc, h_acc, nh_acc

# Get prediction scores (probability estimates)
def get_scores(clf, X):
    return clf.predict_proba(X)[:, 1]