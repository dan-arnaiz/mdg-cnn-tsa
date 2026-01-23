#!/usr/bin/env python3
"""
DDoS Simulation Topology - Automation Optimized
8 hosts, 1 switch, 1 Ryu controller (OpenFlow 1.3)
"""
import time
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

    # Add all 8 hosts
    hosts = []
    for i in range(1, 9):
        hosts.append(net.addHost(f'h{i}', ip=f'10.0.0.{i}/24'))

    # Link all hosts to switch
    for h in hosts:
        net.addLink(h, s1, bw=100, delay='5ms')

    net.build()
    c0.start()
    s1.start([c0])

    print("=========================================")
    print("*** TOPOLOGY ACTIVE ***")
    print("=========================================")
    
    # KEEP ACTIVE: This prevents the 'No such file or directory' namespace error
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()