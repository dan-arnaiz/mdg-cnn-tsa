#!/bin/bash
# Automated DDoS Test Runner
# Runs benign traffic, then DDoS attack, then analyzes results

VICTIM="10.0.0.1"
BENIGN_DURATION=30
ATTACK_DURATION=30
COOLDOWN=10

echo "=========================================="
echo "DDoS Detection Test - Automated Runner"
echo "=========================================="
echo ""

# Phase 1: Benign Traffic Only
echo "[Phase 1] Generating benign traffic ($BENIGN_DURATION seconds)..."
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Start benign hosts
mininet> h2 ./benign_traffic.sh $VICTIM $BENIGN_DURATION &
mininet> h3 ./benign_traffic.sh $VICTIM $BENIGN_DURATION &
mininet> h4 ./benign_traffic.sh $VICTIM $BENIGN_DURATION &
mininet> h5 ./benign_traffic.sh $VICTIM $BENIGN_DURATION &

echo "Benign traffic started from h2, h3, h4, h5..."
sleep $((BENIGN_DURATION + 5))

echo ""
echo "[Phase 1] Complete - Benign traffic finished"
echo ""
sleep $COOLDOWN

# Phase 2: DDoS Attack
echo "=========================================="
echo "[Phase 2] Launching DDoS attack ($ATTACK_DURATION seconds)..."
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Start attackers
mininet> h6 ./ddos_attack.sh $VICTIM $ATTACK_DURATION 2000 &
mininet> h7 ./ddos_attack.sh $VICTIM $ATTACK_DURATION 2000 &
mininet> h8 ./ddos_attack.sh $VICTIM $ATTACK_DURATION 2000 &

echo "DDoS attack started from h6, h7, h8..."
echo "Attack rate: ~6000 packets/second total"
sleep $((ATTACK_DURATION + 5))

echo ""
echo "[Phase 2] Complete - Attack finished"
echo ""

# Phase 3: Analysis
echo "=========================================="
echo "[Phase 3] Analyzing results..."
echo "=========================================="
echo ""

if [ -f "merged_outputs/detections.log" ]; then
    TOTAL=$(wc -l < merged_outputs/detections.log)
    DDOS=$(awk -F',' '$2 >= 0.5' merged_outputs/detections.log | wc -l)
    BENIGN=$(awk -F',' '$2 < 0.5' merged_outputs/detections.log | wc -l)
    
    echo "Total predictions: $TOTAL"
    echo "DDoS detected: $DDOS"
    echo "Benign: $BENIGN"
    echo ""
    echo "Detection rate: $(awk "BEGIN {printf \"%.2f\", ($DDOS/$TOTAL)*100}")%"
else
    echo "No detections.log found!"
fi

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "Check merged_outputs/detections.log for detailed results"