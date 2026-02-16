#!/usr/bin/env python3
"""
DDoS Simulation Topology
8 hosts, 1 switch, Remote Ryu controller (OpenFlow 1.3)
Automation-friendly (keeps Mininet alive)
"""

import time
import sys
import signal
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel, info
from mininet.link import TCLink

net = None

def cleanup(signum=None, frame=None):
    global net
    if net is not None:
        info('\n*** Stopping Mininet\n')
        net.stop()
    sys.exit(0)

def run_automation_mode():
    global net

    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        autoStaticArp=True
    )

    info('*** Adding controller\n')
    c0 = net.addController(
        name='c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653
    )

    info('*** Adding switch\n')
    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    info('*** Adding hosts\n')
    h1 = net.addHost('h1', ip='10.0.0.1/24')  # Victim

    hosts = []
    for i in range(2, 9):
        hosts.append(net.addHost(f'h{i}', ip=f'10.0.0.{i}/24'))

    info('*** Adding links\n')
    for h in [h1] + hosts:
        net.addLink(h, s1, bw=100, delay='5ms')

    info('*** Building network\n')
    net.build()

    info('*** Starting controller\n')
    c0.start()

    info('*** Starting switch\n')
    s1.start([c0])

    info('*** Mininet is READY (automation mode)\n')

    # Signal readiness
    with open('/tmp/mininet_ready', 'w') as f:
        f.write('READY\n')

    print("TOPOLOGY_READY")
    sys.stdout.flush()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

if __name__ == '__main__':
    setLogLevel('info')
    run_automation_mode()
