#!/usr/bin/env python3
"""
DDoS Simulation Topology
8 hosts, 1 switch, 1 Ryu controller (OpenFlow 1.3)
"""
import time
import signal
import sys
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel
from mininet.link import TCLink

def run():
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    c0 = net.addController(
        name='c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6653
    )

    s1 = net.addSwitch('s1', protocols='OpenFlow13')

    # Victim
    h1 = net.addHost('h1', ip='10.0.0.1/24')

    # Benign and Attackers
    hosts = []
    for i in range(2, 9):
        hosts.append(net.addHost(f'h{i}', ip=f'10.0.0.{i}/24'))

    # Link all hosts to switch
    for h in [h1] + hosts:
        net.addLink(h, s1, bw=100, delay='5ms')

    net.build()
    c0.start()
    s1.start([c0])

    print("*** Topology is now active. ***")
    
    # Keep the script running so namespaces stay alive for master.sh
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()