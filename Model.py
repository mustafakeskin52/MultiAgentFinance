from osbrain import Agent
from MessageType import MessageType
from osbrain import run_agent
from osbrain import run_nameserver
class Model(Agent):

    uniqueId = ""
    dataMemory = []
    behaviourState = 0

    def on_init_agent(self,server,connectionFunction):
        self.bind('PUSH', alias='main')
        self.connect(server.addr('main'), handler=connectionFunction)

    def connect_to_new_agent(self,connectionAgent,connectionFunction):
        self.connect(connectionAgent.addr('main'),handler = connectionFunction)

    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message)

    def sending_message(self,sendingObject):
       if sendingObject != None:
            sendingObjectC = MessageType()
            sendingObjectC.__dict__ = sendingObject.__dict__.copy()
            self.send('main', sendingObjectC)
       else:
           self.send('main', None)
    def get_datamemory(self):
        return self.dataMemory
    def get_behaviourstate(self):
        return self.behaviourState