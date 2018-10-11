from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
from MessageType import MessageType

class Server(Agent):

    def on_init(self):
        self.bind('PUB', alias='main')

    def server_broadcast(self,broad_casting_object):
        if broad_casting_object != None:
            sendingObjectC = MessageType()
            sendingObjectC.__dict__ = broad_casting_object.__dict__.copy()
            self.send('main', sendingObjectC)
        else:
            self.send('main', None)