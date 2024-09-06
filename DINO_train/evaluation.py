from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
import numpy as np

def evaluate_prediction(scores, labels, reversed=False):
    combined = list(zip(scores, labels))
    combined = sorted(combined, key=lambda x: x[0], reverse=reversed)
    combined = np.array(combined)

    for i in range(combined.shape[0] + 1):
        ls, rs = i, combined.shape[0] - i
        preds = np.concatenate((np.zeros(ls), np.ones(rs)))
        recall = recall_score(combined[:, 1], preds, pos_label=0)
        if recall >= 0.95:
            return preds, combined[:, 1]
    
    return None, combined[:, 1]  

def evaluate(scores, labels, reversed=False):
    """Requires lower score to mean more likely to be non-hybrid,
    and higher score to mean more likely to be hybrid.
    
    If you would like this to be reversed, set reversed=True
    """
    preds, gt = evaluate_prediction(scores, labels, reversed)

    if preds is None:
        return None  
    
    h_recall = recall_score(gt, preds, pos_label=1)
    h_precision = precision_score(gt, preds, pos_label=1)
    f1 = f1_score(gt, preds, pos_label=1)
    roc_auc = roc_auc_score(gt, preds)
    acc = accuracy_score(gt, preds)

    return h_recall, h_precision, f1, roc_auc, acc

def print_evaluation(h_recall, h_precision, f1, roc_auc, acc):
    print(f"""
          Hybrid-Recall: {h_recall}
          Hybrid-Precision: {h_precision}
          f1-Score: {f1}
          ROC AUC: {roc_auc}
          Accuracy: {acc}
          """)

def print_major_minor_stats(scores, labels, camids, major_cams, minor_cams, reversed=False):
    preds, gt = evaluate_prediction(scores, labels, reversed=reversed)
    tmp = list(zip(scores, labels, camids))
    tmp = sorted(tmp, key=lambda x: x[0], reverse=reversed)
    sorted_camids = np.array(tmp)[:, 2]
    major_idx = np.isin(np.array(sorted_camids), major_cams)
    minor_idx = np.isin(np.array(sorted_camids), minor_cams)
    maj_acc = accuracy_score(gt[major_idx], preds[major_idx])
    min_acc = accuracy_score(gt[minor_idx], preds[minor_idx])
    sub_acc = accuracy_score(gt[~np.logical_or(minor_idx, major_idx)], preds[~np.logical_or(minor_idx, major_idx)])
    sub_recall = recall_score(gt, preds, pos_label=0)
    print(f"Subspecies Classifier (SC) | Major Acc: {maj_acc} | Minor Acc: {min_acc} | Subspecies Acc: {sub_acc} | {sub_recall}")
