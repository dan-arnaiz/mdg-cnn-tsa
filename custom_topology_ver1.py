from mininet.topo import Topo

class MyTopo( Topo ):
	"DDOS detection topology testbed ver 1"
	def build(self):
		server1 = self.addHost('Server1',ip='10.0.0.100/24')
		server2 = self.addHost('Server2',ip='10.0.0.101/24')
		switch1 = self.addSwitch('S1')
		switch2 = self.addSwitch('S2')
		switch3 = self.addSwitch('S3')
		switch4 = self.addSwitch('S4')
		host1 = self.addHost('H1')
		host2 = self.addHost('H2')
		host3 = self.addHost('H3')
		host4 = self.addHost('H4')

		self.addLink(host1,switch1)
		self.addLink(host2,switch1)
		self.addLink(host3,switch2)
		self.addLink(host4,switch2)
		self.addLink(switch1,switch3)
		self.addLink(switch2,switch3)
		self.addLink(switch3,switch4)
		self.addLink(switch4,server1)
		self.addLink(switch4,server2)

topos = {'mytopo': (lambda:MyTopo())}
