#!/usr/bin/env python3
"""
DDoS Simulation Topology
8 hosts, 1 switch, 1 Ryu controller (OpenFlow 1.3)
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
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

    # Benign
    h2 = net.addHost('h2', ip='10.0.0.2/24')
    h3 = net.addHost('h3', ip='10.0.0.3/24')
    h4 = net.addHost('h4', ip='10.0.0.4/24')
    h5 = net.addHost('h5', ip='10.0.0.5/24')

    # Attackers
    h6 = net.addHost('h6', ip='10.0.0.6/24')
    h7 = net.addHost('h7', ip='10.0.0.7/24')
    h8 = net.addHost('h8', ip='10.0.0.8/24')

    for h in [h1,h2,h3,h4,h5,h6,h7,h8]:
        net.addLink(h, s1, bw=100, delay='5ms')

    net.build()
    c0.start()
    s1.start([c0])

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()
