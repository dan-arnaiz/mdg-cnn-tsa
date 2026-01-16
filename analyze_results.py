#!/usr/bin/env python3
"""
DDoS Detection Results Analyzer
Analyzes the detections.log file and generates statistics
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_detections(log_file):
    """Analyze detection results and generate statistics"""
    
    timestamps = []
    predictions = []
    
    print("=" * 70)
    print("DDoS DETECTION ANALYSIS")
    print("=" * 70)
    print()
    
    # Read log file
    try:
        with open(log_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    ts, pred = float(parts[0]), float(parts[1])
                    timestamps.append(ts)
                    predictions.append(pred)
    except FileNotFoundError:
        print(f"Error: {log_file} not found!")
        return
    
    if not predictions:
        print("No predictions found in log file!")
        return
    
    predictions = np.array(predictions)
    timestamps = np.array(timestamps)
    
    # Calculate statistics
    total = len(predictions)
    ddos_detected = np.sum(predictions >= 0.5)
    benign = total - ddos_detected
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    max_pred = np.max(predictions)
    min_pred = np.min(predictions)
    
    # Time range
    start_time = datetime.fromtimestamp(timestamps[0])
    end_time = datetime.fromtimestamp(timestamps[-1])
    duration = timestamps[-1] - timestamps[0]
    
    # Print statistics
    print("OVERALL STATISTICS:")
    print("-" * 70)
    print(f"Total Predictions:     {total}")
    print(f"DDoS Detected:         {ddos_detected} ({ddos_detected/total*100:.2f}%)")
    print(f"Benign:                {benign} ({benign/total*100:.2f}%)")
    print()
    print(f"Mean Prediction:       {mean_pred:.4f}")
    print(f"Std Deviation:         {std_pred:.4f}")
    print(f"Max Prediction:        {max_pred:.4f}")
    print(f"Min Prediction:        {min_pred:.4f}")
    print()
    print(f"Start Time:            {start_time}")
    print(f"End Time:              {end_time}")
    print(f"Duration:              {duration:.2f} seconds")
    print()
    
    # Time-series analysis
    print("TIME-SERIES ANALYSIS:")
    print("-" * 70)
    
    # Divide into windows
    window_size = 10  # seconds
    num_windows = int(duration / window_size) + 1
    
    for i in range(num_windows):
        window_start = timestamps[0] + i * window_size
        window_end = window_start + window_size
        
        mask = (timestamps >= window_start) & (timestamps < window_end)
        window_preds = predictions[mask]
        
        if len(window_preds) > 0:
            window_ddos = np.sum(window_preds >= 0.5)
            window_total = len(window_preds)
            window_rate = window_ddos / window_total * 100
            
            status = "ATTACK" if window_rate > 50 else "NORMAL"
            print(f"Window {i+1:2d} ({window_size*i:3d}-{window_size*(i+1):3d}s): "
                  f"{window_total:3d} flows, {window_ddos:3d} DDoS ({window_rate:5.1f}%) [{status}]")
    
    print()
    print("=" * 70)
    
    # Generate visualization
    generate_plots(timestamps, predictions)

def generate_plots(timestamps, predictions):
    """Generate visualization plots"""
    
    # Normalize timestamps to start from 0
    rel_timestamps = timestamps - timestamps[0]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Prediction scores over time
    axes[0].plot(rel_timestamps, predictions, 'b-', linewidth=0.5, alpha=0.6)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    axes[0].fill_between(rel_timestamps, 0, predictions, 
                         where=(predictions >= 0.5), 
                         color='red', alpha=0.3, label='DDoS')
    axes[0].fill_between(rel_timestamps, 0, predictions, 
                         where=(predictions < 0.5), 
                         color='green', alpha=0.3, label='Benign')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Prediction Score')
    axes[0].set_title('DDoS Detection Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Detection histogram
    axes[1].hist(predictions, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Prediction Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: detection_results.png")
    print()

if __name__ == '__main__':
    log_file = 'merged_outputs/detections.log' if len(sys.argv) < 2 else sys.argv[1]
    analyze_detections(log_file)