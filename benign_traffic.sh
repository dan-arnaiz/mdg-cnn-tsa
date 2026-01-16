#!/bin/bash
# Benign Traffic Generator
# Normal web browsing simulation

TARGET=$1
DURATION=${2:-60}

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds]"
    exit 1
fi

echo "Starting benign traffic to $TARGET for $DURATION seconds..."

END=$((SECONDS+DURATION))

while [ $SECONDS -lt $END ]; do
    # HTTP-like requests (80-120 bytes)
    ping -c 1 -s 100 $TARGET > /dev/null 2>&1
    sleep 0.$(shuf -i 100-500 -n 1)  # Random delay 0.1-0.5s
    
    # Simulate different packet sizes
    SIZE=$(shuf -i 64-1500 -n 1)
    ping -c 1 -s $SIZE $TARGET > /dev/null 2>&1
    sleep 0.$(shuf -i 200-800 -n 1)  # Random delay 0.2-0.8s
done

echo "Benign traffic completed"