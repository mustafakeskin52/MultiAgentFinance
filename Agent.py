import time
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent

class Server(Agent):
    def on_init(self):
        self.bind('PUB', alias='main')

    def server_broadcast(self,broadcastingObject):
        self.send('main', broadcastingObject)


class MessageType:
    message = ""
    senderId = ""


class Model(Agent):
    uniqueId = ""

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
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer)

    def sending_message(self,sendingObject):
       if sendingObject != None:
            sendingObjectC = MessageType()
            sendingObjectC.__dict__ = sendingObject.__dict__.copy()
            self.send('main', sendingObjectC)
       else:
           self.send('main', None)

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

if __name__ == '__main__':

    modelsList = []
    # System deployment
    ns = run_nameserver()
    model1 = run_agent('Model1', base=Model)
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

    for _ in range(3):

        #example code
       # server.server_broadcast(p1.uniqueId)

        sendingObjectList = {"model1": m1, "model2": None, "model3": m1}
        m1.message = 5
        m1.senderId = "model1"
        communicateALLAgents(modelsList, m1.senderId,sendingObjectList)
        m1.message = 7
        m1.senderId = "model2"
        communicateALLAgents(modelsList,  m1.senderId, sendingObjectList)
        time.sleep(1)

    ns.shutdown()