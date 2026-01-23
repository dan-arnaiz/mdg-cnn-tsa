#!/bin/bash
# Enhanced Benign Traffic Generator
# Simulates Mixed Web (HTTP), DNS (UDP), and ICMP traffic

TARGET=$1
DURATION=${2:-60}

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds]"
    exit 1
fi

echo "Starting high-fidelity benign traffic to $TARGET..."
END=$((SECONDS+DURATION))

while [ $SECONDS -lt $END ]; do
    # 1. Simulate ICMP (Standard background noise)
    ping -c 1 -s $(shuf -i 64-128 -n 1) $TARGET > /dev/null 2>&1
    
    # 2. Simulate HTTP-style TCP Bursts (using curl if available, or hping3)
    # This creates short-lived TCP flows that stay under your pkt_rate threshold.
    if command -v curl &> /dev/null; then
        curl -s --connect-timeout 1 http://$TARGET > /dev/null 2>&1
    else
        # Small TCP SYN packet (normal connection attempt)
        hping3 -S -p 80 -c 2 $TARGET > /dev/null 2>&1
    fi

    # 3. Simulate UDP (DNS/Video small packets)
    # Sends 1-5 random UDP packets
    hping3 -2 -p 53 -c $(shuf -i 1-5 -n 1) $TARGET > /dev/null 2>&1

    # Random "Human" delay between 0.5s and 2.0s
    sleep $(echo "scale=2; $(shuf -i 50-200 -n 1)/100" | bc)
done

echo "Benign traffic completed"