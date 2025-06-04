from sklearn import metrics
import numpy as np


def evaluation_metrics(gt_flat: np.ndarray, detection_map_flat: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Computes AUC-based evaluation metrics from ROC curves.

    This function calculates multiple Area Under the Curve (AUC) metrics to evaluate
    anomaly detection performance. It derives the True Positive Rate (TPR) and
    False Positive Rate (FPR) using a Receiver Operating Characteristic (ROC) curve.

    Args:
        gt_flat (np.ndarray): Flattened ground truth binary mask (shape: [N]), where 1 indicates
                              an anomaly and 0 represents the background.
        detection_map_flat (np.ndarray): Flattened detection scores (shape: [N]), where higher
                                         values indicate higher anomaly likelihood. Values are [0,1]

    Returns:
        tuple[float, float, float, float, float]:
            - auc_effect (float): AUC(Pf, Pd) - Standard ROC-AUC score.
            - auc_false_alarm (float): AUC(τ, Pf) - Measures how false alarms vary with the threshold.
            - auc_detect (float): AUC(τ, Pd) - Measures how detection rate varies with the threshold.
            - auc_oa (float): Overall accuracy metric (AUC_effect + AUC_detect - AUC_false_alarm).
            - auc_snpr (float): Signal-to-Noise Power Ratio (AUC_detect / AUC_false_alarm).

    Raises:
        ValueError: If `gt_flat` contains values other than 0 and 1.
        ValueError: If `gt_flat` and `detection_map_flat` have different shapes.
    """
    if not np.all(np.isin(gt_flat, [0, 1])):
        raise ValueError("Ground truth array must only contain binary values (0 and 1).")

    if gt_flat.shape != detection_map_flat.shape:
        raise ValueError("Ground truth and detection map must have the same shape.")

    # Compute ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(gt_flat, detection_map_flat, drop_intermediate=False)

    # Exclude the first value to avoid threshold -inf
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    auc_effect = round(metrics.auc(fpr, tpr), 5)  # AUC(Pf, Pd)
    auc_false_alarm = round(metrics.auc(thresholds, fpr), 5)  # AUC(τ, Pf)
    auc_detect = round(metrics.auc(thresholds, tpr), 5)  # AUC(τ, Pd)
    auc_oa = round(auc_effect + auc_detect - auc_false_alarm, 5)  # AUC Overall Accuracy
    auc_snpr = round(auc_detect / auc_false_alarm, 5) if auc_false_alarm != 0 else float(
        "inf")  # Avoid division by zero

    return auc_effect, auc_detect, auc_false_alarm, auc_oa, auc_snpr
