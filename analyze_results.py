#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

DETECTION_THRESHOLD = 0.50


def load_detection_data(log_file):
    timestamps = []
    predictions = []
    true_labels = []
    packet_rates = []

    try:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')

                # EXPECT 5 COLUMNS NOW
                if len(parts) == 5:
                    ts, pred, pred_label, true_label, pkt_rate = parts
                    timestamps.append(float(ts))
                    predictions.append(float(pred))
                    true_labels.append(int(true_label))
                    packet_rates.append(float(pkt_rate))

    except FileNotFoundError:
        print(f"Error: {log_file} not found!")
        return None, None, None, None

    return (np.array(timestamps),
            np.array(predictions),
            np.array(true_labels),
            np.array(packet_rates))


def calculate_metrics(y_true, y_pred_proba):

    # Threshold classification (DO NOT invert)
    y_pred = (y_pred_proba >= DETECTION_THRESHOLD).astype(int)

    # Compute confusion matrix safely
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case if only one class exists
        tn = fp = fn = tp = 0
        if y_true[0] == 0:
            tn = cm[0][0]
        else:
            tp = cm[0][0]

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # ROC-AUC (correct probability, no inversion)
    if len(np.unique(y_true)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr_curve, tpr_curve)
    else:
        fpr_curve, tpr_curve, roc_auc = None, None, 0.5

    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "fpr_curve": fpr_curve,
        "tpr_curve": tpr_curve
    }


def plot_confusion_matrix(metrics, path):
    cm = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]]
    ])

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Attack'],
                yticklabels=['Benign','Attack'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc(metrics, path):
    if metrics["fpr_curve"] is None:
        return

    plt.figure(figsize=(6,5))
    plt.plot(metrics["fpr_curve"], metrics["tpr_curve"],
             label=f"AUC={metrics['roc_auc']:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    log_file = "merged_outputs/detections.log"
    timestamps, predictions, labels, pkt = load_detection_data(log_file)

    if timestamps is None or len(predictions) == 0:
        print("No data.")
        return

    metrics = calculate_metrics(labels, predictions)

    print("="*80)
    print("DDOS DETECTION ANALYSIS - CORRECTED REPORT")
    print("="*80)
    print()

    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Total Predictions: {len(predictions)}")
    print(f"Actual DDoS Flows: {np.sum(labels==1)}")
    print(f"Actual Benign Flows: {np.sum(labels==0)}")
    print()

    print("PERFORMANCE METRICS")
    print("-"*80)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1-Score: {metrics['f1']*100:.2f}%")
    print(f"False Positive Rate: {metrics['fpr']*100:.2f}%")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print()

    os.makedirs("merged_outputs", exist_ok=True)
    plot_confusion_matrix(metrics, "merged_outputs/confusion_matrix.png")
    plot_roc(metrics, "merged_outputs/roc_curve.png")

    print("Visualizations saved to merged_outputs/")


if __name__ == "__main__":
    main()