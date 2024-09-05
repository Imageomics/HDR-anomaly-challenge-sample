from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

def train(X, y, classifier_config="sgd"):
    non_hybrid_weight = 1
    hybrid_weight = 1
    class_weights = {0: non_hybrid_weight, 1: hybrid_weight}

    if classifier_config == "svm":
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1, class_weight=class_weights, probability=True))
    elif classifier_config == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=0.001,
            penalty="l2",
            eta0=0.001,
            n_iter_no_change=100,
            learning_rate='constant',
            max_iter=10000,
            class_weight=class_weights
        )
    elif classifier_config == "knn":
        clf = KNeighborsClassifier(n_neighbors=2)
    elif classifier_config == "gaussian":
        clf = GaussianProcessClassifier(random_state=0)
    else:
        raise ValueError("Invalid classifier_config")

    clf.fit(X, y)

    preds = clf.predict(X)
    correct = preds == y

    hybrid_correct = correct[y == 1].sum()
    non_hybrid_correct = correct[y == 0].sum()

    acc = clf.score(X, y)
    h_acc = hybrid_correct / (y == 1).sum()
    nh_acc = non_hybrid_correct / (y == 0).sum()

    return clf, acc, h_acc, nh_acc


def get_scores(clf, X):
    return clf.predict_proba(X)[:, 1]