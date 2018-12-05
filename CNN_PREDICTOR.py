from ModelAgent import Model
from Experiment import Experiment
import numpy as np
import config
import model
import time
import dataset
from sklearn import linear_model
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

#A model might extend to the class that is a abstract agent model including basic layouts
class CNN_PREDICTOR(Model):

    lastPrediction = 0
    timeSeriesLength = 20
    model_cnn = None
    model_fit = None
    trainLength = 0
    thresholding = []
    config = config.ConfigCNN()
    experiment = None
    def on_init_properity(self,trainLength,thresholding):
        self.trainLength = trainLength
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
    def train(self,dataN):
        classDatas = self.dataToClassFunc(dataN,self.thresholding)
        data = dataset.CNN1DOnlineDataSet(raw_dataset=classDatas,seq_len=self.timeSeriesLength)
        self.model_lstm = model.CNN(self.config)
        self.experiment = Experiment(config=self.config, model=self.model_lstm, dataset=data)
        self.experiment.run()
        #print("Actual:",classDatas[self.config.SEQ_LEN+100])
        #print("Predicted:",self.experiment.predict_lstm(classDatas[100:self.config.SEQ_LEN+100],self.config.INPUT_SIZE))
    def predict(self):
        classDatas = self.dataToClassFunc(np.asarray(self.dataMemory[-self.timeSeriesLength:]),self.thresholding)
        return np.asarray(self.experiment.predict_cnn(classDatas))[0]
    def dataToClassFunc(self,data, thresholding):
        result = np.zeros(data.shape[0])
        for i, d in enumerate(data):
            if d > thresholding[0]:
                result[i] = BehaviourState.HIGH_BUY
            elif d > thresholding[1]:
                result[i] = BehaviourState.BUY
            elif d > thresholding[2]:
                result[i] = BehaviourState.NONE
            elif d > thresholding[3]:
                result[i] = BehaviourState.SELL
            else:
                result[i] = BehaviourState.LOW_SELL
        return result
    # The method provides to send to message from self to another agent
    def evaluate_behaviour(self):
        t = self.dataTime[-1]

        if len(self.dataMemory) > self.timeSeriesLength:
            self.behaviourState = self.predict()
        else:
            self.behaviourState = BehaviourState.NONE