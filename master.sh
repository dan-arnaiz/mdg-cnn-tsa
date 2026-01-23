#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection - FULLY AUTOMATED

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
ryu-manager controller.py > controller.log 2>&1 &
RYU_PID=$!
sleep 15 
echo "Ryu Controller Active."
echo "========================================="

# --- 3. Start Mininet Topology ---
echo "========================================="
echo "[3/7] Starting Mininet Topology..."
sudo python3 topology.py > topology.log 2>&1 &
TOPO_PID=$!
sleep 20 # Wait for namespaces to initialize

# --- DYNAMIC NAMESPACE CHECK ---
# This looks at what Linux actually created
NS_PREFIX=""
if sudo ip netns | grep -q "mininet-h2"; then
    NS_PREFIX="mininet-"
    echo "Detected Namespace Prefix: 'mininet-'"
else
    echo "Detected Namespace Prefix: None"
fi
echo "Mininet running in the background."
echo "========================================="

# --- 4. Benign Phase ---
echo "========================================="
echo "[4/7] Phase 1: Benign Traffic (60 seconds)..."
for i in {2..5}; do
    sudo ip netns exec ${NS_PREFIX}h${i} ./benign_traffic.sh 10.0.0.1 60 &
done
sleep 70 

echo "Gap period: Waiting 20 seconds..."
sleep 20
echo "========================================="

# --- 5. Attack Phase ---
echo "========================================="
echo "[5/7] Phase 2: DDoS Attack (60 seconds)..."
# Target h1 (10.0.0.1) with 3 simultaneous attackers
for i in {6..8}; do
    sudo ip netns exec ${NS_PREFIX}h${i} ./ddos_attack.sh 10.0.0.1 60 2000 &
done
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