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
            self.behaviourState = regr.predict(t)
