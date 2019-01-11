import time
import config
import os.path
import pickle
import model
import dataset
from LSTM_PREDICTOR import LSTM_PREDICTOR
import HelperFunctions as hp
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
from Server import Server
from LSTM_DECIDER import LSTM_DECIDER
from Experiment import Experiment
import pandas as pd
import mnist_main
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ARIMAAgent import ARIMAAgent
from CNN_PREDICTOR import CNN_PREDICTOR
from tqdm import trange, tqdm
from MajorityDecider import MajorityDecider
from LinearRegAgent import LinearRegAgent
from EvaluatorAgent import EvaluatorAgent
from movingAverageAgent import MovingAverageAgent
from ImitatorAgent import ImitatorAgent
from MessageType import MessageType
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
from sklearn import linear_model
class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0
def downSampling(signal,periodicDownSampling):
    samplingSignal = []
    for i,d in enumerate(signal):
        if i%periodicDownSampling == 0:
            samplingSignal.append(d)
    return np.asarray(samplingSignal)
def initialize_agent():
    trainingRate = 0.7
    #data = pd.DataFrame(data = np.add(hp.sinData(3000,10), hp.cosData(3000,50)))#np.add(hp.sinData(1000,30), hp.sinData(1000,50)) #np.asarray(hp.sinData(1000,30))#hp.sinData(1000,30)#np.add(hp.sinData(1000,30), hp.sinData(1000,50))
    #s = data.iloc[0:1000]
    downSamplingSignal = downSampling(hp.readDataFromCSV("AMD.CSV")[0:],1)
    s = pd.Series(downSamplingSignal)
    N = 2  # Filter order

    Wn = 0.4
    B, A = signal.butter(N, Wn, output='ba')
    w, h = signal.freqs(B, A)

    #Before this code comed to this block ,the signal was passing to a low pass filter
    s = pd.DataFrame(data = signal.filtfilt(B, A, s))
    plt.plot(s,'b',label='Line 1')
    plt.legend()
    plt.show()
    percentageData = np.asarray(s.pct_change())[1:] * 100
    trainingLength = int(percentageData.shape[0]*trainingRate)
    trainingdata = percentageData[:trainingLength]
    thresholdingVector = hp.findOptimalThresholds(trainingdata, 5)
    #trainingdata = trainingdata.squeeze(axis=1)

    """
        
    """
    #s = pd.Series(hp.readDataFromCSV("AMD.CSV")[9000:9400])
    #s = data[1000:3000]
    testdata = percentageData[trainingLength:]
    #testdata = testdata.squeeze(axis=1)
    # Setting evaluater thresholding vector table so that class loss error will be calculated by the evaluater
    model1 = run_agent('Model1', base=LinearRegAgent)
    model3 = run_agent('Model3', base=MovingAverageAgent)
    evaluate_agent = run_agent('Evaluator', base=EvaluatorAgent)
    #cnn_agent = run_agent('CNN_Agent',base = CNN_PREDICTOR)
    lstm_agent = run_agent('LSTM_Agent',base = LSTM_PREDICTOR)
    lstm_decider =run_agent('LSTM_DECIDER',base=LSTM_DECIDER)
    imitator = run_agent('Imitator', base=ImitatorAgent)
    majorityDecider = run_agent('MajorityDecider', base=MajorityDecider)
    arimaAgent = run_agent('ARIMAAgent', base=ARIMAAgent)

    model1.uniqueId = "model1"
    model3.uniqueId = "model3"
    #cnn_agent.uniqueId = "cnn_agent"
    arimaAgent.uniqueId = "arima"
    evaluate_agent.uniqueId = "evaluater"
    imitator.uniqueId = "imitator"
    lstm_agent.uniqueId = "lstm_agent"
    majorityDecider.uniqueId = "majorityDecider"
    lstm_decider.uniqueId = "lstm_decider"

    #cnn_agent.on_init_properity(3,thresholdingVector)
    model1.on_init_properity(3,thresholdingVector)
    model3.on_init_properity(5,thresholdingVector)
    lstm_agent.on_init_properity(None,thresholdingVector)
    arimaAgent.on_init_properity(5,thresholdingVector)
    evaluate_agent.on_init_properity(thresholdingVector)
    lstm_decider.on_init_properity(3,thresholdingVector)

    #cnn_agent.train(trainingdata)
    lstm_agent.train(trainingdata)

    modelsList.append(model1)
    modelsList.append(lstm_agent)
    modelsList.append(arimaAgent)
    #modelsList.append(cnn_agent)
    modelsList.append(evaluate_agent)
    modelsList.append(imitator)
    modelsList.append(majorityDecider)
    modelsList.append(lstm_decider)

    return modelsList,testdata
if __name__ == '__main__':
    modelsList = []
    filePath = "globalsave"
    ns = run_nameserver()
    server = run_agent('Server', base=Server)

    config = config.ConfigLSTM()

    modelsList, data = initialize_agent()

    hp.initialConnectionsAgent(modelsList,server)
    # Send messages
    m1 = MessageType()
    #modelsList = loadDatas(modelsList,filePath)
    hp.loadDatas(modelsList,filePath)
    agentlist = hp.getAgentList(modelsList)#all agent names have been stored in this list

    lstmdeciderLog = []
    majorityVotingLog = []
    for i,d in enumerate(data):
        #In the loop for testing some probabilities
        majorityDeciderFeedBack = []
        m1.message = [d, i]
        """
            Server broadcast the message that have been taking lately from database or real dataset  
        """
        server.server_broadcast(m1)

        print(len(data))
        """
            After server broadcasting the message and all messages had been reached to agents by server,the evaluater will be updated because of given reasons:
                *Before agents will publish to their behaviours,evaluator should calculate their scores and return to these scores as object to server
                *The priority of the calculating scores can be important due to server priority that depended on agents based system 
        """
        modelsList[3].update()
        print("time:",i)
        print("data:",d)
        """
            In this part of the agent.py code main loop after evaluater collects behaviours of the agents ,they decided to broadcast it to imitator.
            Therefore,at the first part of the code might be sended to a empty behaviours to the agents  
        """
        sendingObjectList = {"model1": None,"lstm_agent": None,"arima":None,"evaluater": None, "imitator": m1,"majorityDecider":None,"lstm_decider":None}
        m1.message = modelsList[3].getLastBehavioursAgents()
        # # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourTruthLast"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        """
            Agent behaviours are calculating in this scope.If agent will be added as finding possible solution of financial trending ,
            the for each loop range will be increased or changed.
            After passing this block,all the agents can explain their behaviours to server. 
        """
        for j in range(0,3,1):
            modelsList[j].evaluate_behaviour()

        """
            All decisions is sending to evaluater before running majoritydecider or any decider agent.
        """
        sendingObjectList = {"model1": None,"lstm_agent": None,"arima":None,"evaluater": m1,
                             "imitator": None, "majorityDecider": None,"lstm_decider":None}
        for j in range(0,3,1):
            m1.message = modelsList[j].get_behaviourstate()
            m1.senderId = modelsList[j].uniqueId
            m1.messageType = "behaviourOfAgentNow"
            majorityDeciderFeedBack.append(m1.message)
            hp.communicateALLAgents(modelsList,m1.senderId,sendingObjectList)
        """
            Evaluator is sending their datas to majoritydecider to take a greater score than all agents behaviours 
        """
        sendingObjectList = {"model1": None,"lstm_agent": None,"arima":None,"evaluater": None, "imitator": None,
                             "majorityDecider": m1,"lstm_decider":m1}
        m1.message = majorityDeciderFeedBack
        m1.senderId = "evaluater"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        #Decider will be runned in this method because of its priority
        #This priority is more important than some priority that is located in this while loop
        modelsList[-1].evaluate_behaviour()
        modelsList[-2].evaluate_behaviour()
        """
            After majoritydecider published its decision,it is sending majority decider discrete from all general agents
            The purpose of  the majority decider algoritms must obtain a greater score than all agents.Hence,evaluater agents will be 
            argumentative. 
        """
        sendingObjectList = {"model1": None,"lstm_agent": None,"arima":None,"evaluater": m1, "imitator": None,
                             "majorityDecider": None,"lstm_decider":None}

        m1.message = modelsList[-1].get_behaviourstate()
        m1.senderId = modelsList[-1].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        m1.message = modelsList[-2].get_behaviourstate()
        m1.senderId = modelsList[-2].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        """
            Imitator object might be considered as a lstm that try to improve all agents performences.
            The differences between imitator and decider agents is that lstm might be more successful than classical decider according to our plan
        """
        sendingObjectList = {"model1": None,"lstm_agent": None,"arima":None,"evaluater": None, "imitator": m1,"majorityDecider": None,"lstm_decider":None}
        m1.message = modelsList[3].getAgentLastPredictionList()
        # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        print("GeneralScores:",modelsList[3].getAgentScores())
        print("PeriodicScores:",modelsList[3].getPeriodicScoreTableAgents())

        if modelsList[3].getPeriodicScoreTableAgents() != {}:
            lstmdeciderLog.append(modelsList[3].getPeriodicScoreTableAgents()["lstm_decider"])
            majorityVotingLog.append(modelsList[3].getPeriodicScoreTableAgents()["majorityDecider"])
        #print("PeriodicDatas",modelsList[3].getPeriodicScoreTableAgents())
    #hp.saveDatas(modelsList, filePath)
    plt.plot(lstmdeciderLog,'r',label='LstmDecider')  # plotting t, a separately
    plt.plot(majorityVotingLog,'b',label='majorityVoting')  # plotting t, b separately
    plt.legend()
    plt.show()

    #modelsList[5].saveDataFrameCSV()
    ns.shutdown()
