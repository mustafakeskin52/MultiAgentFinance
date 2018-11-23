from ModelAgent import Model
import numpy as np
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class BehaviourState:
    BUY = 1
    SELL = -1
    NONE = 0

#A model might extend to class that is a abstract agent model including basic layouts
class ARIMAAgent(Model):

    lastPrediction = 0
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
    # The method provides to send to message from self to another agent
    def evaluate_behaviour(self,lastN):
        t = self.dataTime[-1]

        if len(self.dataMemory) >= lastN:
            model = ARIMA(self.dataMemory[0:t+1], order=(0, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            predictionValue = output[0]
            if (predictionValue - self.lastPrediction)> 0:
                self.behaviourState = BehaviourState.BUY
            else:
                self.behaviourState = BehaviourState.SELL
            self.lastPrediction = predictionValue