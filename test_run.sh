#!/bin/bash
# test_run.sh - Automated testing with ground truth tracking

echo "=== DDoS Detection Test Run ==="
echo "Start time: $(date)"

# Clear old logs
rm -f merged_outputs/detections.log

# Phase 1: Benign (60s)
echo "[Phase 1] Benign traffic starting..."
echo "0,60,benign" > merged_outputs/ground_truth.txt

# Wait 65 seconds
sleep 65

# Phase 2: Attack (60s) 
echo "[Phase 2] DDoS attack starting..."
echo "65,125,attack" >> merged_outputs/ground_truth.txt

# Wait 65 seconds
sleep 65

echo "Test complete!"
echo "Run: python analyze_results.py"