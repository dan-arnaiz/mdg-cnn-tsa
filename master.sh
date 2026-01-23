#!/bin/bash
# Master Test Script for CNN-TSA DDoS Detection - FULLY AUTOMATED
# Usage: ./master.sh

set -e  # Exit on error
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup_all() {
    log_warn "Cleaning up all processes..."
    sudo pkill -9 -f ryu-manager 2>/dev/null || true
    sudo pkill -9 -f topology.py 2>/dev/null || true
    sudo pkill -9 -f python3.*topology 2>/dev/null || true
    sudo mn -c 2>/dev/null || true
    sudo killall -9 ping 2>/dev/null || true
    sudo killall -9 hping3 2>/dev/null || true
    rm -f /tmp/mininet_ready 2>/dev/null || true
}

# Trap for cleanup on exit
trap cleanup_all EXIT

# --- PHASE 0: Preparation ---
echo "========================================="
log_info "Phase 0: Preparation"
echo "========================================="

# Cleanup old processes and logs
cleanup_all
rm -rf merged_outputs/*
mkdir -p merged_outputs

# Make scripts executable
chmod +x benign_traffic.sh ddos_attack.sh topology.py analyze_results.py

log_info "Cleanup complete."
echo ""

# --- PHASE 1: Start Ryu Controller ---
echo "========================================="
log_info "Phase 1: Starting Ryu Controller"
echo "========================================="

ryu-manager controller.py > merged_outputs/controller.log 2>&1 &
RYU_PID=$!

log_info "Waiting for controller initialization (15 seconds)..."
sleep 15

# Verify controller is running
if ps -p $RYU_PID > /dev/null; then
    log_info "Ryu Controller is running (PID: $RYU_PID)"
else
    log_error "Controller failed to start! Check merged_outputs/controller.log"
    exit 1
fi

echo ""
echo "========================================="
log_info "Phase 2: Starting Mininet"
echo "========================================="

sudo python3 topology.py > merged_outputs/topology.log 2>&1 &
TOPO_PID=$!

TIMEOUT=30
ELAPSED=0

while [ ! -f /tmp/mininet_ready ] && [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ ! -f /tmp/mininet_ready ]; then
    log_error "Mininet failed to start"
    exit 1
fi

sleep 5

log_info "Verifying network namespaces..."

if sudo ip netns exec mn-h2 ip addr show > /dev/null 2>&1; then
    log_info "Network namespaces verified"
else
    log_error "Mininet namespaces NOT found"
    sudo ip netns list
    exit 1
fi

echo ""

# --- PHASE 3: Benign Traffic ---
echo "========================================="
log_info "Phase 3: Benign Traffic (60 seconds)"
echo "========================================="

echo "$(date +%s),benign_start,Benign traffic phase started" > merged_outputs/test_timeline.txt

log_info "Starting benign traffic from h2, h3, h4, h5..."

# Launch benign traffic
for i in {2..5}; do
    sudo ip netns exec mn-h${i} bash -c "cd $SCRIPT_DIR && ./benign_traffic.sh 10.0.0.1 60" > /dev/null 2>&1 &
    BENIGN_PIDS[$i]=$!
    log_info "  h${i}: Started (PID: ${BENIGN_PIDS[$i]})"
done

log_info "Benign traffic running. Waiting 70 seconds..."

# Wait with progress indicator
for i in {1..70}; do
    echo -ne "\rProgress: [$i/70] seconds elapsed"
    sleep 1
done
echo ""

echo "$(date +%s),benign_end,Benign traffic phase completed" >> merged_outputs/test_timeline.txt
log_info "Benign phase complete"

# Gap period
log_info "Gap period: Waiting 20 seconds for traffic to settle..."
sleep 20

echo ""

# --- PHASE 4: DDoS Attack ---
echo "========================================="
log_info "Phase 4: DDoS Attack (60 seconds)"
echo "========================================="

echo "$(date +%s),attack_start,DDoS attack phase started (6000 pps total)" >> merged_outputs/test_timeline.txt

log_info "Launching DDoS attack from h6, h7, h8..."

# Launch attack traffic
for i in {6..8}; do
    sudo ip netns exec mn-h${i} bash -c "cd $SCRIPT_DIR && ./ddos_attack.sh 10.0.0.1 60 2000" > /dev/null 2>&1 &
    ATTACK_PIDS[$i]=$!
    log_info "  h${i}: Attack started (PID: ${ATTACK_PIDS[$i]})"
done

log_info "Attack in progress. Waiting 70 seconds..."

# Wait with progress indicator
for i in {1..70}; do
    echo -ne "\rProgress: [$i/70] seconds elapsed"
    sleep 1
done
echo ""

echo "$(date +%s),attack_end,DDoS attack phase completed" >> merged_outputs/test_timeline.txt
log_info "Attack phase complete"

# Additional wait for final flow stats
log_info "Waiting 10 seconds for final flow statistics..."
sleep 10

echo ""

# --- PHASE 5: Analysis ---
echo "========================================="
log_info "Phase 5: Generating Analysis Report"
echo "========================================="

if [ -f merged_outputs/detections.log ]; then
    LOG_SIZE=$(wc -l < merged_outputs/detections.log)
    log_info "Detection log contains $LOG_SIZE entries"
    
    if [ $LOG_SIZE -gt 0 ]; then
        python3 analyze_results.py | tee merged_outputs/analysis_report.txt
        log_info "Analysis complete"
    else
        log_warn "Detection log is empty! No analysis performed."
        log_warn "Check merged_outputs/controller.log for issues"
    fi
else
    log_error "Detection log not found!"
    log_error "Controller may not have processed any flows"
fi

echo ""

# --- PHASE 6: Cleanup ---
echo "========================================="
log_info "Phase 6: Final Cleanup"
echo "========================================="

log_info "Stopping Ryu controller (PID: $RYU_PID)..."
sudo kill -TERM $RYU_PID 2>/dev/null || true

log_info "Stopping Mininet topology (PID: $TOPO_PID)..."
sudo kill -TERM $TOPO_PID 2>/dev/null || true

sleep 2

log_info "Cleaning up network..."
sudo mn -c > /dev/null 2>&1 || true

echo ""

# --- PHASE 7: Summary ---
echo "========================================="
log_info "Test Complete - Summary"
echo "========================================="

echo ""
echo "Generated Files:"
echo "  - merged_outputs/detections.log       (Raw detection data)"
echo "  - merged_outputs/analysis_report.txt  (Full analysis)"
echo "  - merged_outputs/metrics_summary.txt  (Quick metrics)"
echo "  - merged_outputs/confusion_matrix.png (Visualization)"
echo "  - merged_outputs/roc_curve.png        (ROC curve)"
echo "  - merged_outputs/time_series.png      (Time series plot)"
echo "  - merged_outputs/controller.log       (Controller logs)"
echo "  - merged_outputs/topology.log         (Topology logs)"
echo ""

if [ -f merged_outputs/metrics_summary.txt ]; then
    echo "Quick Metrics:"
    cat merged_outputs/metrics_summary.txt
else
    log_warn "Metrics summary not generated"
fi

echo ""
log_info "All done! Check merged_outputs/ for detailed results."
echo "========================================="