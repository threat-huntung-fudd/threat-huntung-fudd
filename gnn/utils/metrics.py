import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_link_prediction_metrics(
    predicts: torch.Tensor,
    labels: torch.Tensor,
    structural_mask: torch.Tensor,
    contextual_mask: torch.Tensor,
    temporal_mask: torch.Tensor,
):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :param structural_mask: Tensor, shape (num_samples, )
    :param contextual_mask: Tensor, shape (num_samples, )
    :param temporal_mask: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    structural_mask = structural_mask.cpu().numpy()
    contextual_mask = contextual_mask.cpu().numpy()
    temporal_mask = temporal_mask.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    if len(np.unique(labels)) <= 1:
            f1 = -1
    else:
            y_pred_binary = (predicts > 0.5).astype(int)
            f1 = f1_score(y_true=labels, y_pred=y_pred_binary)

    structural_roc_auc = (
        roc_auc_score(y_true=labels[structural_mask], y_score=predicts[structural_mask])
        if not np.all(structural_mask == False)
        else 0.50
    )
    contextual_roc_auc = (
        roc_auc_score(y_true=labels[contextual_mask], y_score=predicts[contextual_mask])
        if not np.all(contextual_mask == False)
        else 0.50
    )

    temporal_roc_auc = (
        roc_auc_score(y_true=labels[temporal_mask], y_score=predicts[temporal_mask])
        if not np.all(temporal_mask == False)
        else 0.50
    )

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "structural_roc_auc": structural_roc_auc,
        "contextual_roc_auc": contextual_roc_auc,
        "temporal_roc_auc": temporal_roc_auc,
        "f1": f1,
    }


def get_link_prediction_metrics_classification(
    predicts: torch.Tensor,
    labels: torch.Tensor,
):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :param structural_mask: Tensor, shape (num_samples, )
    :param contextual_mask: Tensor, shape (num_samples, )
    :param temporal_mask: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    y_pred_class = np.argmax(predicts, axis=1)
    labels = labels.cpu().numpy()

    # average_precision = average_precision_score(y_true=labels, y_score=predicts)
    # roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    accuracy = accuracy_score(y_true=labels, y_pred=y_pred_class)
    precision = precision_score(y_true=labels, y_pred=y_pred_class, average='weighted')
    recall = recall_score(y_true=labels, y_pred=y_pred_class, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=y_pred_class, average='weighted')   

#   "average_precision": average_precision,
# "roc_auc": roc_auc,
    return {      
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
