#!/usr/bin/env python3
"""
DDoS Detection Results Analyzer
Analyze detections.log and generate required metrics.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def load_detection_data(log_file):
    """Load detection results from log file"""
    timestamps = []
    predictions = []
    labels = []
    packet_rates = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    ts, pred, label, pkt_rate = parts
                    timestamps.append(float(ts))
                    predictions.append(float(pred))
                    labels.append(int(label))
                    packet_rates.append(float(pkt_rate))
    except FileNotFoundError:
        print(f"Error: {log_file} not found!")
        return None, None, None, None
    
    return (np.array(timestamps), np.array(predictions), 
            np.array(labels), np.array(packet_rates))


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate all classification metrics"""
    
    # Binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Confusion matrix components
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # ROC-AUC
    if len(np.unique(y_true)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr_curve, tpr_curve)
    else:
        fpr_curve, tpr_curve, roc_auc = None, None, 0.0
    
    metrics = {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': fpr,
        'roc_auc': roc_auc,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve
    }
    
    return metrics


def analyze_time_series(timestamps, predictions, labels, window_size=10):
    """Analyze detection performance over time windows"""
    
    if len(timestamps) == 0:
        return []
    
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time
    
    num_windows = int(duration / window_size) + 1
    windows = []
    
    for i in range(num_windows):
        window_start = start_time + i * window_size
        window_end = window_start + window_size
        
        mask = (timestamps >= window_start) & (timestamps < window_end)
        
        if np.sum(mask) > 0:
            window_preds = predictions[mask]
            window_labels = labels[mask]
            window_ddos = np.sum(window_preds >= 0.5)
            window_actual_ddos = np.sum(window_labels == 1)
            window_total = len(window_preds)
            
            detection_rate = (window_ddos / window_total * 100) if window_total > 0 else 0
            actual_rate = (window_actual_ddos / window_total * 100) if window_total > 0 else 0
            
            status = "ATTACK" if detection_rate > 50 else "NORMAL"
            
            windows.append({
                'window': i + 1,
                'start': window_start - start_time,
                'end': window_end - start_time,
                'total': window_total,
                'detected_ddos': window_ddos,
                'actual_ddos': window_actual_ddos,
                'detection_rate': detection_rate,
                'actual_rate': actual_rate,
                'status': status
            })
    
    return windows


def plot_confusion_matrix(metrics, save_path='confusion_matrix.png'):
    """Generate confusion matrix visualization"""
    cm = np.array([[metrics['TN'], metrics['FP']], 
                   [metrics['FN'], metrics['TP']]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Benign', 'Predicted Attack'],
                yticklabels=['Actual Benign', 'Actual Attack'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(metrics, save_path='roc_curve.png'):
    """Generate ROC curve visualization"""
    if metrics['fpr_curve'] is None:
        print("Cannot plot ROC curve: insufficient data")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr_curve'], metrics['tpr_curve'], 
             color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def plot_time_series(timestamps, predictions, labels, save_path='time_series.png'):
    """Generate time series detection visualization"""
    rel_timestamps = timestamps - timestamps[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Prediction scores over time
    axes[0].plot(rel_timestamps, predictions, 'b-', linewidth=0.8, alpha=0.7, label='Prediction Score')
    axes[0].axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
    
    # Color background based on actual labels
    attack_mask = labels == 1
    benign_mask = labels == 0
    
    axes[0].fill_between(rel_timestamps, 0, 1, where=attack_mask, 
                         color='red', alpha=0.1, label='Actual Attack Period')
    axes[0].fill_between(rel_timestamps, 0, 1, where=benign_mask, 
                         color='green', alpha=0.05, label='Actual Benign Period')
    
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('Prediction Score', fontsize=12)
    axes[0].set_title('DDoS Detection Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])
    
    # Plot 2: Prediction distribution
    axes[1].hist(predictions[benign_mask], bins=50, color='green', alpha=0.6, 
                label=f'Benign (n={np.sum(benign_mask)})', edgecolor='black')
    axes[1].hist(predictions[attack_mask], bins=50, color='red', alpha=0.6, 
                label=f'Attack (n={np.sum(attack_mask)})', edgecolor='black')
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_xlabel('Prediction Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Prediction Scores by Actual Class', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to: {save_path}")


def print_analysis_report(timestamps, predictions, labels, packet_rates, metrics, windows):
    """Print comprehensive analysis report"""
    
    print("=" * 80)
    print("DDOS DETECTION ANALYSIS - COMPREHENSIVE REPORT")
    print("=" * 80)
    print()
    
    # Overall Statistics
    print("OVERALL STATISTICS:")
    print("-" * 80)
    print(f"Total Predictions:     {len(predictions)}")
    print(f"Actual DDoS Flows:     {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.2f}%)")
    print(f"Actual Benign Flows:   {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.2f}%)")
    print()
    print(f"Mean Prediction Score: {np.mean(predictions):.4f}")
    print(f"Std Deviation:         {np.std(predictions):.4f}")
    print(f"Max Prediction:        {np.max(predictions):.4f}")
    print(f"Min Prediction:        {np.min(predictions):.4f}")
    print()
    
    # Time information
    start_time = datetime.fromtimestamp(timestamps[0])
    end_time = datetime.fromtimestamp(timestamps[-1])
    duration = timestamps[-1] - timestamps[0]
    
    print(f"Start Time:            {start_time}")
    print(f"End Time:              {end_time}")
    print(f"Duration:              {duration:.2f} seconds")
    print()
    
    # Confusion Matrix
    print("CONFUSION MATRIX:")
    print("-" * 80)
    print(f"{'':20s} {'Predicted Benign':>20s} {'Predicted Attack':>20s}")
    print(f"{'Actual Benign':20s} {metrics['TN']:>20d} {metrics['FP']:>20d}")
    print(f"{'Actual Attack':20s} {metrics['FN']:>20d} {metrics['TP']:>20d}")
    print()
    
    # Performance Metrics
    print("PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"Accuracy:              {metrics['accuracy']*100:>6.2f}%")
    print(f"Precision:             {metrics['precision']*100:>6.2f}%")
    print(f"Recall (Sensitivity):  {metrics['recall']*100:>6.2f}%")
    print(f"F1-Score:              {metrics['f1_score']*100:>6.2f}%")
    print(f"False Positive Rate:   {metrics['fpr']*100:>6.2f}%")
    print(f"ROC-AUC:               {metrics['roc_auc']:>6.4f}")
    print()
    
    # Time-Series Analysis
    print("TIME-SERIES ANALYSIS (10-second windows):")
    print("-" * 80)
    print(f"{'Window':<8s} {'Time Range':<15s} {'Flows':<8s} {'Detected':<10s} "
          f"{'Actual':<8s} {'Det.Rate':<10s} {'Status':<10s}")
    print("-" * 80)
    
    for w in windows:
        print(f"{w['window']:<8d} "
              f"{int(w['start']):>4d}-{int(w['end']):<4d}s     "
              f"{w['total']:<8d} "
              f"{w['detected_ddos']:<10d} "
              f"{w['actual_ddos']:<8d} "
              f"{w['detection_rate']:>5.1f}%     "
              f"{w['status']:<10s}")
    
    print()
    print("=" * 80)
    print("END OF REPORT")
    print("=" * 80)


def main():
    log_file = 'merged_outputs/detections.log' if len(sys.argv) < 2 else sys.argv[1]
    
    # Load data
    timestamps, predictions, labels, packet_rates = load_detection_data(log_file)
    
    if timestamps is None:
        return
    
    if len(predictions) == 0:
        print("No predictions found in log file!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(labels, predictions)
    
    # Time-series analysis
    windows = analyze_time_series(timestamps, predictions, labels)
    
    # Print report
    print_analysis_report(timestamps, predictions, labels, packet_rates, metrics, windows)
    
    # Generate visualizations
    plot_confusion_matrix(metrics, 'merged_outputs/confusion_matrix.png')
    plot_roc_curve(metrics, 'merged_outputs/roc_curve.png')
    plot_time_series(timestamps, predictions, labels, 'merged_outputs/time_series.png')
    
    print()
    print("All visualizations saved to merged_outputs/")
    
    # Save metrics to file
    with open('merged_outputs/metrics_summary.txt', 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Precision: {metrics['precision']*100:.2f}%\n")
        f.write(f"Recall: {metrics['recall']*100:.2f}%\n")
        f.write(f"F1-Score: {metrics['f1_score']*100:.2f}%\n")
        f.write(f"FPR: {metrics['fpr']*100:.2f}%\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
    
    print("Metrics summary saved to merged_outputs/metrics_summary.txt")


if __name__ == '__main__':
    main()