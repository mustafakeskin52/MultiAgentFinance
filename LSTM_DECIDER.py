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

class LSTM_DECIDER(Model):
    agentsBeheviours = []
    dataX = []
    model_lstm = None
    model_fit = None
    config = config.ConfigLSTMForDecider()
    experiment = None
    trainLength = None
    thresholding = None
    startPointOfTraining = 300
    periodOfTraining = 50
    def on_init_properity(self, trainLength, thresholding):
        self.trainLength = trainLength
        self.thresholding = thresholding
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.agentsBeheviours = receivingObjectFromAgent.message

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
    def train(self,dataX,dataY):
        data = dataset.OnlineDeciderDataSet(seq_len=self.config.SEQ_LEN, raw_dataset_x=dataX,raw_dataset_y=dataY)
        self.model_lstm = model.LSTM(input_size=self.config.INPUT_SIZE, seq_length=self.config.SEQ_LEN, num_layers=2,
                          out_size=self.config.OUTPUT_SIZE, hidden_size=5, batch_size=self.config.TRAIN_BATCH_SIZE,
                           device=self.config.DEVICE)
        self.experiment = Experiment(config=self.config, model=self.model_lstm, dataset=data)
        self.experiment.run()
        #print("Predicted:",self.experiment.predict_lstm(classDatas[100:self.config.SEQ_LEN+100],self.config.INPUT_SIZE))
    def predict(self,dataX):
        return np.asarray(self.experiment.predict_lstm_decider(dataX[-self.config.SEQ_LEN:],5))[0]
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
        self.agentsBeheviours.append(int(classData[-1]))#Original Increasing Class is being added to lstm input
        print("self.agentsBeheviours",self.agentsBeheviours)
        self.dataX.append(self.agentsBeheviours)

        if (len(self.dataMemory) >= self.startPointOfTraining and len(self.dataMemory) % self.periodOfTraining == 0):
            self.train(np.asarray(self.dataX), classData)
        if len(self.dataMemory) > self.startPointOfTraining:
            self.behaviourState = self.predict(np.asarray(self.dataX))
        print("lstm_decider",self.behaviourState)

