from ModelAgent import Model
import numpy as np
from sklearn import linear_model
class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

class MajorityDecider(Model):
    agentsBeheviours = []
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.agentsBeheviours = receivingObjectFromAgent.message

    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message[0])
        self.dataMemory.append(receivingObjectFromServer.message[0])
        self.dataTime.append(receivingObjectFromServer.message[1])

    def loadALLVariables(self, pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.dataMemory = data['dataMemory'].tolist()
        self.dataTime = data['dataTime'].tolist()

    def saveALLVariables(self, pathOfImitatorObject):
        np.savez(pathOfImitatorObject,dataMemory=self.dataMemory,
                 dataTime=self.dataTime)
    # The method provide to send to message from self to another agent
    def evaluate_behaviour(self):
        t = self.dataTime[-1]
        print("self.agentsBeheviours",self.agentsBeheviours)
        lastBehaviour = np.asarray(self.agentsBeheviours)
        unique_elements, counts_elements = np.unique(lastBehaviour, return_counts=True)
        self.behaviourState = unique_elements[np.argmax(counts_elements)]

