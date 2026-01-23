#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection Thesis

# --- 1. Cleanup ---
echo "========================================="
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f mininet
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs
echo "Cleanup complete."
echo "========================================="

# --- 2. Start Ryu Controller ---
echo "========================================="
echo "[2/7] Starting Ryu Controller..."
# Controller logs 39 features and hidden_dim=64
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 15 
echo "Model Successfully Loaded into Ryu Controller."
echo "========================================="

# --- 3. Start Mininet Topology ---
echo "========================================="
echo "[3/7] Starting Mininet Topology..."
# Note: Ensure CLI(net) is commented out in topology.py for full automation
sudo python3 topology.py & 
sleep 20 
echo "Mininet Topology is up and running."
echo "========================================="

# --- 4. Benign Phase ---
echo "========================================="
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."
# Executing within the specific network namespaces of your 8 hosts
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
# Attackers h6, h7, and h8 targeting the victim h1 (10.0.0.1)
sudo ip netns exec mininet-h6 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec mininet-h7 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo ip netns exec mininet-h8 ./ddos_attack.sh 10.0.0.1 60 2000 &
sleep 80 
echo "Done with DDoS Attack Phase."
echo "========================================="

# --- 6. Analysis and Reporting ---
echo "========================================="
echo "[6/7] Generating Comprehensive Report..."
# analyze_results.py uses the 0.40 threshold for correct metrics
sudo python3 analyze_results.py | tee merged_outputs/live_report.txt
echo "Report successfully generated."
echo "========================================="

# --- 7. Final Cleanup ---
echo "========================================="
echo "[7/7] Experiment Complete."
sudo kill $RYU_PID
sudo mn -c > /dev/null 2>&1
echo "========================================="
echo "All processes cleaned up. Exiting."