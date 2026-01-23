#!/usr/bin/env python3
"""
DDoS Simulation Topology
8 hosts, 1 switch, 1 Ryu controller (OpenFlow 1.3)
Stays active for external command execution
"""
import time
import sys
import signal
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

# Global network object
net = None

def cleanup(signum=None, frame=None):
    """Cleanup handler for signals"""
    global net
    if net is not None:
        info('\n*** Stopping network\n')
        net.stop()
    sys.exit(0)

def run_automation_mode():
    """Run topology in automation mode - stays alive for external commands"""
    global net
    
    net = Mininet(
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
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
    # Victim
    h1 = net.addHost('h1', ip='10.0.0.1/24')

    # Benign hosts (h2-h5) and Attackers (h6-h8)
    hosts = []
    for i in range(2, 9):
        hosts.append(net.addHost(f'h{i}', ip=f'10.0.0.{i}/24'))

    info('*** Adding links\n')
    # Link all hosts to switch
    for h in [h1] + hosts:
        net.addLink(h, s1, bw=100, delay='5ms')

    info('*** Building network\n')
    net.build()
    
    info('*** Starting controller\n')
    c0.start()
    
    info('*** Starting switch\n')
    s1.start([c0])

    info('*** Network is ready\n')
    info('*** Topology running in automation mode...\n')
    
    # Write ready signal
    with open('/tmp/mininet_ready', 'w') as f:
        f.write('READY\n')
    
    print("TOPOLOGY_READY")  # Signal for master.sh
    sys.stdout.flush()
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Keep network alive - wait for termination signal
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()

def run_interactive_mode():
    """Run topology in interactive CLI mode"""
    global net
    
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

    info('*** Running CLI\n')
    CLI(net)
    
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    
    # Check if running in automation mode
    if '--automation' in sys.argv or '-a' in sys.argv:
        run_automation_mode()
    else:
        run_interactive_mode()