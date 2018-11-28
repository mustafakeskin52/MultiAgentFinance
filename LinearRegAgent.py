from ModelAgent import Model
import numpy as np
from sklearn import linear_model

class BehaviourState:
    HIGH_BUY = 2
    BUY = 1
    NONE = 0
    SELL = -1
    LOW_SELL = -2


#A model might extend to class that is a abstract agent model including basic layouts
class LinearRegAgent(Model):
    lastN = 0
    thresholding = []

    def on_init_properity(self,lastN,thresholding):
        self.lastN = lastN
        self.thresholding = thresholding
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)

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
        t = self.dataTime[-1]+1
        time = np.arange(t - self.lastN, t, 1)
        time = time.reshape(-1, 1)

        if len(self.dataMemory) >= self.lastN:
            #print(t)
            regr = linear_model.LinearRegression()
            regr.fit(time,self.dataMemory[t - self.lastN:t])
            predictionValue = regr.predict(t)

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