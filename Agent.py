import time
import numpy as np
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
from sklearn import linear_model


class MessageType:
    message = 56
    senderId = ""

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

class Model(Agent):
    uniqueId = ""
    dataMemory = []
    behaviourState = 0

    def on_init_agent(self,server,connectionFunction):
        self.bind('PUSH', alias='main')
        self.connect(server.addr('main'), handler=connectionFunction)

    def connect_to_new_agent(self,connectionAgent,connectionFunction):
        self.connect(connectionAgent.addr('main'),handler = connectionFunction)
    #
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


#A model might extend to class that is a abstract agent model including basic layouts
class LinearRegAgent(Model):

    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)

    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message)
        self.dataMemory.append(receivingObjectFromServer.message)

    def evaluate_behaviour(self):
        lastN = 3
        t = len(self.dataMemory)
        time = np.arange(t - lastN, t, 1)
        time = time.reshape(-1, 1)
        if t > lastN:
            regr = linear_model.LinearRegression()
            regr.fit(time,self.dataMemory[t - lastN:t])
            predictionValue = regr.predict(t)

            if (predictionValue - self.dataMemory[t-1])> 0:
                self.behaviourState = 1
            else:
                self.behaviourState = -1

def initialConnectionsAgent(modelsList):
    # if agent receive a message from server,code will flow to the connectionFunction.
    # Therefore the parameter must been given truthly to could run the code

    for i in range(len(modelsList)):
        modelsList[i].on_init_agent(server, 'receive_server_broadcast_message')

    # to use this function for the purpose of connecting a model to another model
    # one agent is being a listener and another agent is being a publisher
    # model1.connect_to_new_agent(model2,'receive_message')
    # model2.connect(model2.addr('main'),handler = 'receive_message')

    for i in range(len(modelsList)):
        for j in range(len(modelsList)):
            if i != j:
                modelsList[i].connect_to_new_agent(modelsList[j], 'receive_agent_message')
def communicateALLAgents(modelsList,sendingAgent,sendingObjectLists):
    modelIndex = 0
    for index in range(len(modelsList)):
        if modelsList[index].uniqueId == sendingAgent:
            modelIndex = index;
            break;
    for key in sendingObjectLists:
        if sendingAgent != key:
            if  sendingObjectLists[key] != None:
                modelsList[modelIndex].sending_message(sendingObjectLists[key])
            else:
                modelsList[modelIndex].sending_message(None)
def getAgentList(modelLists):
    agentLists = []
    for i in range(len(modelsList)):
        agentLists.append(modelsList[i].uniqueId)
    return agentLists
if __name__ == '__main__':

    modelsList = []
    # System deployment
    ns = run_nameserver()
    model1 = run_agent('Model1', base=LinearRegAgent)
    model2 = run_agent('Model2', base=Model)
    model3 = run_agent('Model3',base=Model)
    server = run_agent('Server', base=Server)

    model1.uniqueId = "model1"
    model2.uniqueId = "model2"
    model3.uniqueId = "model3"

    modelsList.append(model1)
    modelsList.append(model2)
    modelsList.append(model3)

    initialConnectionsAgent(modelsList)
    # Send messages
    m1 = MessageType()

    agentlist = getAgentList(modelsList)#all agent names have been stored in this list

    for _ in range(30):

        m1.message = 5
        server.server_broadcast(m1)

        modelsList[0].evaluate_behaviour()
        print(modelsList[0].get_behaviourstate())

        #example code
        #sendingObjectList = {"model1": m1, "model2": None, "model3": m1}
        #m1.message = 5
        #m1.senderId = "model1"
        #communicateALLAgents(modelsList, m1.senderId,sendingObjectList)
        #m1.message = 7
        #m1.senderId = "model2"
        #communicateALLAgents(modelsList,  m1.senderId, sendingObjectList)
        #time.sleep(1)
    ns.shutdown()