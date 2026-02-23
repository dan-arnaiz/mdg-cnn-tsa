#!/usr/bin/env python3
"""
DDoS Simulation Topology
8 hosts, 1 switch, Remote Ryu controller (OpenFlow 1.3)
Fully automation-safe for master.sh
"""

import os
import time
import sys
import signal
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.clean import cleanup as mn_cleanup

READY_FILE = "/tmp/mininet_ready"

net = None


# ─────────────────────────────────────────────────────────────
# Clean shutdown
# ─────────────────────────────────────────────────────────────
def cleanup(signum=None, frame=None):
    global net
    info("\n*** Cleaning up topology\n")

    try:
        if net is not None:
            net.stop()
    except Exception:
        pass

    try:
        if os.path.exists(READY_FILE):
            os.remove(READY_FILE)
    except Exception:
        pass

    mn_cleanup()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# Automation mode
# ─────────────────────────────────────────────────────────────
def run_automation_mode():
    global net

    info("*** Creating network\n")

    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True,
        build=False
    )

    # Controller
    info("*** Adding remote controller (Ryu at 127.0.0.1:6653)\n")
    c0 = net.addController(
        name="c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6653
    )

    # Switch
    info("*** Adding switch s1 (OpenFlow13)\n")
    s1 = net.addSwitch(
        "s1",
        protocols="OpenFlow13",
        failMode="secure"
    )

    # Victim
    info("*** Adding victim host h1 (10.0.0.1)\n")
    h1 = net.addHost("h1", ip="10.0.0.1/24")

    # Other hosts
    info("*** Adding attacker/benign hosts h2–h8\n")
    hosts = []
    for i in range(2, 9):
        host = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        hosts.append(host)

    # Links
    info("*** Creating links\n")
    for h in [h1] + hosts:
        net.addLink(
            h,
            s1,
            bw=100,
            delay="5ms",
            loss=0
        )

    # Build network
    info("*** Building network\n")
    net.build()

    info("*** Starting controller\n")
    c0.start()

    info("*** Starting switch\n")
    s1.start([c0])

    # Wait briefly for controller handshake
    info("*** Waiting for switch-controller handshake...\n")
    time.sleep(3)

    # Confirm switch connected
    if not s1.connected():
        info("*** WARNING: Switch not yet connected to controller\n")
    else:
        info("*** Switch connected to controller\n")

    # Signal readiness to master.sh
    with open(READY_FILE, "w") as f:
        f.write("READY\n")

    print("TOPOLOGY_READY")
    sys.stdout.flush()

    info("*** Topology ready for automation\n")

    # Keep alive for automation
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setLogLevel("info")

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Ensure no stale ready file exists
    if os.path.exists(READY_FILE):
        os.remove(READY_FILE)

    run_automation_mode()