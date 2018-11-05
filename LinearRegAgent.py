from ModelAgent import Model
import numpy as np
from sklearn import linear_model

class BehaviourState:
    BUY = 1
    SELL = -1
    NONE = 0


#A model might extend to class that is a abstract agent model including basic layouts
class LinearRegAgent(Model):

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
    def evaluate_behaviour(self,lastN):
        t = self.dataTime[-1]+1
        time = np.arange(t - lastN, t, 1)
        time = time.reshape(-1, 1)

        if len(self.dataMemory) >= lastN:
            #print(t)
            regr = linear_model.LinearRegression()
            regr.fit(time,self.dataMemory[t - lastN:t])
            predictionValue = regr.predict(t)

            if (predictionValue - self.dataMemory[t-1])> 0:
                self.behaviourState = BehaviourState.BUY
            else:
                self.behaviourState = BehaviourState.SELL