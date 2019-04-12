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

        lastBehaviour = np.asarray(self.agentsBeheviours)

        result = np.count_nonzero(lastBehaviour > 0) / lastBehaviour.size
        if result > 0.5:
            self.behaviourState = 1
        else:
            self.behaviourState = -1

