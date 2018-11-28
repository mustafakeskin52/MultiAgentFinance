from osbrain import Agent
from MessageType import MessageType
import numpy as np
from osbrain import run_agent
from osbrain import run_nameserver

#The model is  a abstract class that is writed by aid of osbrain library
#And some methods are writed hardly to run server with multi agents
class Model(Agent):

    uniqueId = ""
    dataMemory = []
    dataClassMemory = []
    dataTime = []
    thresholdArray = []
    behaviourState = 0
    #PULL-PUSH relationship communication is actualized with this function

    def on_init_agent(self,server,connectionFunction):
        self.bind('PUSH', alias='main')
        self.connect(server.addr('main'), handler=connectionFunction)

    #If a agent want to connect to the new agent,it must call to the function
    #After it called this function,osbrain library will allow to connect to each others

    def connect_to_new_agent(self,connectionAgent,connectionFunction):
        self.connect(connectionAgent.addr('main'),handler = connectionFunction)

    #This method might be overrided when child class is writing
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)

    #This method might be overrided when child class is writing
    def setThresholding(self,thresholdingArray):
        self.thresholdArray = thresholdingArray
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message[0])
        self.dataMemory.append(receivingObjectFromServer.message[0])

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