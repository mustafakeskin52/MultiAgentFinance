from ModelAgent import Model
import numpy as np
from sklearn import linear_model

class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

#A model might extend to class that is a abstract agent model including basic layouts
class CopyYesterdayAgent(Model):
    thresholding = []

    def on_init_properity(self,thresholding):
        self.thresholding = thresholding
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)

    def loadALLVariables(self, pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.dataMemory = data['dataMemory'].tolist()
        self.dataTime = data['dataTime'].tolist()

    def saveALLVariables(self, pathOfImitatorObject):
        np.savez(pathOfImitatorObject,dataMemory=self.dataMemory,
                 dataTime=self.dataTime)
    # The method provide to send to message from self to another agent
    def evaluate_behaviour(self):
       
        if len(self.dataMemory) >= 1:
            #print(t)
            predictionValue = self.dataMemory[-1]

            if predictionValue > self.thresholding[0]:
                self.behaviourState = BehaviourState.HIGH_BUY
            elif predictionValue > self.thresholding[1]:
                self.behaviourState = BehaviourState.BUY
            elif predictionValue > self.thresholding[2]:
                self.behaviourState = BehaviourState.NONE
            elif predictionValue > self.thresholding[3]:
                self.behaviourState = BehaviourState.SELL
            else:
                self.behaviourState = BehaviourState.LOW_SELL