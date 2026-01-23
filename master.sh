#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection 

# --- 1. Cleanup ---
echo "========================================="
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f topology.py
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs
echo "Cleanup complete."
echo "========================================="

# --- 2. Start Ryu Controller ---
echo "========================================="
echo "[2/7] Starting Ryu Controller..."
# Loading model with 39 features
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 15 
echo "Ryu Controller Active."
echo "========================================="


# --- 3. Start Mininet Topology ---
echo "========================================="
echo "[3/7] Starting Mininet Topology..."
sudo python3 topology.py & 
TOPO_PID=$!
sleep 20 # Essential: Wait for namespaces to initialize
echo "Mininet Topology Active in background."
echo "========================================="


# --- 4. Benign Phase ---
echo "========================================="
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."

# Using standard host names for namespaces
sudo ip netns exec h2 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec h3 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec h4 ./benign_traffic.sh 10.0.0.1 60 &
sudo ip netns exec h5 ./benign_traffic.sh 10.0.0.1 60 &
sleep 70 

echo "Gap period: Waiting 20 seconds..."
sleep 20
echo "========================================="


# --- 5. Attack Phase ---
echo "========================================="

echo "[5/7] Phase 2: DDoS Attack (60 seconds)..."
# High-intensity flood targeting 10.0.0.1
sudo ip netns exec h6 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec h7 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec h8 ./ddos_attack.sh 10.0.0.1 60 2000 &
sleep 80 
echo "Traffic Phases Complete."
echo "========================================="


# --- 6. Analysis and Reporting ---
echo "========================================="

echo "[6/7] Generating Comprehensive Report..."
# Synced threshold 0.40 catches detections peaking at 0.42
sudo python3 analyze_results.py | tee merged_outputs/live_report.txt
echo "========================================="


# --- 7. Final Cleanup ---
echo "========================================="

echo "[7/7] Experiment Complete. Cleaning up..."
sudo kill $RYU_PID
sudo kill $TOPO_PID
sudo mn -c > /dev/null 2>&1
echo "All processes cleaned up. Check merged_outputs/ for results."
echo "========================================="
