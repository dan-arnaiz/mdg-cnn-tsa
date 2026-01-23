#!/bin/bash


# --- 1. Cleanup and Initialization ---
echo "Cleaning up old logs and processes..."
sudo pkill -9 -f ryu-manager
sudo pkill -9 -f mininet
sudo mn -c
rm -rf merged_outputs/*; mkdir -p merged_outputs
/dev/null 2>&1

echo "Done Cleaning up."
