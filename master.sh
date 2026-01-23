#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection Thesis

# --- 1. Cleanup and Initialization ---
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f mininet
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs > /dev/null 2>&1

# --- 2. Start Ryu Controller ---
echo "[2/7] Starting Ryu Controller with CNN-TSA model..."
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 10 # Wait for model to load

# --- 3. Start Mininet Topology ---
echo "[3/7] Starting Mininet Topology..."
# Replace 'topology.py' with your actual topology filename
sudo python3 topology.py & 
sleep 15 # Wait for network stabilization

# --- 4. Benign Phase ---
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."
# Using h2-h5 to generate background noise
sudo mn -v h2 ./benign_traffic.sh 10.0.0.1 60 &
sudo mn -v h3 ./benign_traffic.sh 10.0.0.1 60 &
sudo mn -v h4 ./benign_traffic.sh 10.0.0.1 60 &
sudo mn -v h5 ./benign_traffic.sh 10.0.0.1 60 &
sleep 70 # Buffer time for completion

echo "Gap period: Waiting 20 seconds..."
sleep 20

# --- 5. Attack Phase ---
echo "[5/7] Phase 2: DDoS Attack (60 seconds)..."
# Using h6-h8 for high-intensity attack
sudo mn -v h6 ./ddos_attack.sh 10.0.0.1 60 &
sudo mn -v h7 ./ddos_attack.sh 10.0.0.1 60 &
sudo mn -v h8 ./ddos_attack.sh 10.0.0.1 60 &
sleep 75 # Buffer time to ensure logs are written

# --- 6. Analysis and Reporting ---
echo "[6/7] Generating Comprehensive Report..."
# Ensure analyze_results.py uses the 0.40 threshold
sudo python3 analyze_results.py

# --- 7. Final Cleanup ---
echo "[7/7] Experiment Complete. Cleaning up..."
sudo kill $RYU_PID
sudo mn -c > /dev/null 2>&1
echo "Results available in merged_outputs/ folder."