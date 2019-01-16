from ModelAgent import Model
from Experiment import Experiment
import numpy as np
import model
import dataset
import config
from sklearn import linear_model
class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

class MLPDecider(Model):
    agentsBeheviours = []
    dataX = []
    model_lstm = None
    model_fit = None
    config = config.ConfigMLP()
    experiment = None
    trainLength = None
    thresholding = None
    startPointOfTraining = 100
    periodOfTraining = 50
    def on_init_properity(self, trainLength, thresholding):
        self.trainLength = trainLength
        self.thresholding = thresholding
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
    def train(self,dataX,dataY):
        data = dataset.MLPOnlineDataset(dataX=dataX,dataY=dataY)
        self.model_lstm = model.MLP(input_size=5,output_size=5)
        self.experiment = Experiment(config=self.config, model=self.model_lstm, dataset=data)
        self.experiment.run()
        #print("Predicted:",self.experiment.predict_lstm(classDatas[100:self.config.SEQ_LEN+100],self.config.INPUT_SIZE))
    def predict(self,dataX):
        return np.asarray(self.experiment.predict_mlp_decider(dataX[-30:]))[-1]
    # The method provide to send to message from self to another agent
    def dataToClassFunc(self, data, thresholding):
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
    def evaluate_behaviour(self):
        t = self.dataTime[-1]
        classData = self.dataToClassFunc(np.asarray(self.dataMemory), self.thresholding)
        print("self.agentsBeheviours",self.agentsBeheviours)
        self.dataX.append(self.agentsBeheviours)
        print("dataTypeOfX",np.asarray(self.dataX).shape)
        if (len(self.dataMemory) >= self.startPointOfTraining and len(self.dataMemory) % self.periodOfTraining == 0):
            self.train(np.asarray(self.dataX), classData)
        if len(self.dataMemory) > self.startPointOfTraining:
            self.behaviourState = self.predict(np.asarray(self.dataX))
        print("mlp_decider_state",self.behaviourState)

