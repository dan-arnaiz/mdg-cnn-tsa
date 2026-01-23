#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection Thesis - FIXED

# --- 1. Cleanup ---
echo "========================================="
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f mininet
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs
echo "Done Cleaning up."
echo "========================================="

echo "========================================="
# --- 2. Start Ryu Controller ---
echo "[2/7] Starting Ryu Controller..."
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 15 # Wait for model to load
echo "Model Succesfully Loaded into Ryu Controller."
echo "========================================="



# --- 3. Start Mininet Topology ---
echo "========================================="
echo "[3/7] Starting Mininet Topology..."
# Running in background so the script can continue
sudo python3 topology.py & 
sleep 20 # Wait for switches to connect and STP to settle
echo "Mininet Topology is up and running."
echo "========================================="



# --- 4. Benign Phase ---
echo "========================================="
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."
# Using 'm' command to execute on specific hosts
sudo h2 ./benign_traffic.sh 10.0.0.1 60 &
sudo h3 ./benign_traffic.sh 10.0.0.1 60 &
sudo h4 ./benign_traffic.sh 10.0.0.1 60 &
sudo h5 ./benign_traffic.sh 10.0.0.1 60 &
sleep 70 

echo "Gap period: Waiting 20 seconds..."
sleep 20
echo "Done with Benign Traffic Phase."
echo "========================================="


# --- 5. Attack Phase ---
echo "========================================="
echo "[5/7] Phase 2: DDoS Attack (60 seconds)..."
sudo h6 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo h7 ./ddos_attack.sh 10.0.0.1 60 2000 &
sudo h8 ./ddos_attack.sh 10.0.0.1 60 2000 &
sleep 80 
echo "Done with DDoS Attack Phase."
echo "========================================="

# --- 6. Analysis and Reporting ---
echo "========================================="
echo "[6/7] Generating Comprehensive Report..."
sudo python3 analyze_results.py
echo "Report successfully generated."
echo "========================================="

# --- 7. Final Cleanup ---
echo "========================================="
echo "[7/7] Experiment Complete."
sudo kill $RYU_PID
sudo mn -c > /dev/null 2>&1
echo "========================================="
echo "All processes cleaned up. Exiting."