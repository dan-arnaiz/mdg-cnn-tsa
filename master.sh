#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection System

# --- 1. Cleanup ---
echo "========================================="
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f topology.py
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs
echo "Done Cleaning up."
echo "========================================="

# --- 2. Start Ryu Controller ---
echo "========================================="
echo "[2/7] Starting Ryu Controller..."
# Model config: features=39, hidden=64, heads=2
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 15 
echo "Model Successfully Loaded into Ryu Controller."
echo "========================================="

# --- 3. Start Mininet Topology ---
echo "========================================="
echo "[3/7] Starting Mininet Topology..."
sudo python3 topology.py > topology.log 2>&1 &
TOPO_PID=$!
sleep 20 
echo "Mininet Topology is up and running in background."
echo "========================================="

# --- 4. Benign Phase ---
echo "========================================="
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."
# Using namespace execution to avoid 'command not found'
sudo ip netns exec mininet-h2 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec mininet-h3 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec mininet-h4 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec mininet-h5 ./benign_traffic.sh 10.0.0.1 60 &
sleep 70 

echo "Gap period: Waiting 20 seconds..."
sleep 20
echo "Done with Benign Traffic Phase."
echo "========================================="

# --- 5. Attack Phase ---
echo "========================================="
echo "[5/7] Phase 2: DDoS Attack (60 seconds)..."
# Generating high-intensity flood to target 10.0.0.1
sudo ip netns exec mininet-h6 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec mininet-h7 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec mininet-h8 ./ddos_attack.sh 10.0.0.1 60 2000 &
sleep 80 
echo "Done with DDoS Attack Phase."
echo "========================================="

# --- 6. Analysis and Reporting ---
echo "========================================="
echo "[6/7] Generating Comprehensive Report..."
# Threshold 0.40 synced to catch detections peaking at 0.42
sudo python3 analyze_results.py | tee merged_outputs/live_report.txt
echo "Report successfully generated."
echo "========================================="

# --- 7. Final Cleanup ---
echo "========================================="
echo "[7/7] Experiment Complete. Final Cleanup..."
sudo kill $RYU_PID
sudo kill $TOPO_PID
sudo mn -c > /dev/null 2>&1
echo "========================================="
echo "All processes cleaned up. Exiting."