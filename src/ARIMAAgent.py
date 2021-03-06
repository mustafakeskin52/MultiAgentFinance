from ModelAgent import Model
import numpy as np
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

#A model might extend to class that is a abstract agent model including basic layouts
class ARIMAAgent(Model):

    lastPrediction = 0
    model = None
    model_fit = None
    trainLength = 0
    thresholding = []
    def on_init_properity(self,trainLength,thresholding):
        self.trainLength = trainLength
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
    # The method provides to send to message from self to another agent
    def evaluate_behaviour(self):
        t = self.dataTime[-1]
        print("lastDataArima",self.dataMemory[-1])
        if len(self.dataMemory)%1 == 0 and len(self.dataMemory) > self.trainLength:
            self.model = ARIMA(self.dataMemory[0:t + 1], order=(0, 1, 0))
            self.model_fit = self.model.fit(disp=0)
        if len(self.dataMemory) > self.trainLength:
            output = self.model_fit.forecast()
            self.behaviourState = output[0]
