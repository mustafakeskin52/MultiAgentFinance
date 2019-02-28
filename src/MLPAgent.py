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
class MLPAgent(Model):

    lastPrediction = 0
    model_mlp = None
    model_fit = None
    trainLength = 0
    thresholding = []
    config = config.ConfigMLP()
    experiment = None
    filterSize = 15
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
    def featureExtraction(self,dataN):
        featureMatrix = []

        for i in range(len(dataN)):
            temp = []
            tempFrame = dataN[i:i+self.filterSize]
            tempFrame = np.squeeze(tempFrame, axis=1)
            if i+self.filterSize < len(dataN):
                """Some of the features are being implemented to train a multi layer perceptron"""
                temp.append(tempFrame[0])
                temp.append(tempFrame[-1])
                temp.append(tempFrame[int(self.filterSize/2)])
                temp.append(np.sum(tempFrame >= 0))
                temp.append(np.sum(tempFrame < 0))
                temp.append(np.var(tempFrame))
                temp.append(np.mean(tempFrame))
                temp.append(np.sum(tempFrame[-7:] >= 0))
                temp.append(np.sum(tempFrame[-7:] < 0))
                temp.append(np.sum(tempFrame[-3:] >= 0))
                temp.append(np.sum(tempFrame[-3:] < 0))

            else:
                break
            featureMatrix.append(temp)
        return np.asarray(featureMatrix)
    def train(self,dataN):
        dataY = self.dataToClassFunc(dataN[self.filterSize:], self.thresholding)
        dataX = self.featureExtraction(dataN)

        print("dataX",dataX.shape)
        print("dataY",dataY.shape)

        data = dataset.MLPOnlineDataset(dataX=dataX, dataY=dataY)
        self.model_mlp = model.MLP(input_size=11, output_size=5)
        self.experiment = Experiment(config=self.config, model=self.model_mlp, dataset=data)
        self.experiment.run()

    def predict(self,dataX):
        dataX = self.featureExtraction(dataX)
        return np.asarray(self.experiment.predict_mlp_decider(dataX[-30:]))[-1]
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
    # The method provides to send to message from self to another agent
    def evaluate_behaviour(self):

        if len(self.dataMemory) > 30:
            self.behaviourState = self.predict(np.asarray(self.dataMemory)[-30:])
        else:
            self.behaviourState = BehaviourState.NONE