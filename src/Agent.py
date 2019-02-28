import time
import config
import os.path
import pickle
import model
import dataset
from LSTM_PREDICTOR import LSTM_PREDICTOR
from MLPDecider import MLPDecider
from MLPAgentNW import Mlpagentsp
from RSIAgent import RSIAgent
from CopyYesterdayAgent import CopyYesterdayAgent
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
from MLPAgent import MLPAgent
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


def downSampling(signal, periodicDownSampling):
    samplingSignal = []
    for i, d in enumerate(signal):
        if i % periodicDownSampling == 0:
            samplingSignal.append(d)
    return np.asarray(samplingSignal)

def initialize_agent():
    trainingRate = 0.70
    #data = pd.DataFrame(data = np.add(hp.sinData(3000,40),hp.cosData(3000,80)))#pd.DataFrame(data = np.add(hp.sinData(3000,10), hp.cosData(3000,50)))#np.add(hp.sinData(1000,30), hp.sinData(1000,50)) #np.asarray(hp.sinData(1000,30))#hp.sinData(1000,30)#np.add(hp.sinData(1000,30), hp.sinData(1000,50))
    # s = data.iloc[0:1000]

    downSamplingSignal = downSampling(hp.readDataFromCSV("../input/weather_2017.CSV")[0:], 1)
    s = pd.DataFrame(data=downSamplingSignal)#pd.Series(downSamplingSignal)
    #s = pd.rolling_mean(s,30)[40:]

    #s = data
    plt.plot(s, 'r', label='signal')  # plotting t, a separately
    plt.legend()
    plt.show()
    #s = pd.rolling_mean(pd.Series(downSamplingSignal[0:]),15)[20:]
    #s = pd.Series(downSamplingSignal)
    N = 2  # Filter order

    Wn = 0.3
    B, A = signal.butter(N, Wn, output='ba')

    # Before this code comed to this block ,the signal was passing to a low pass filter
    originalsignal = s

    trainingLength = int(s.shape[0] * trainingRate)

    #s = pd.DataFrame(data=signal.filtfilt(B, A, s[0:trainingLength],axis=0))

    #s = pd.DataFrame(data=s[0:trainingLength])
    #If received data is not as defined a pandas frame,this data must be converted to pandas frame
    #s = pd.rolling_mean(s[0:trainingLength],15)[20:trainingLength]

    trainingData = np.asarray(s.pct_change())[1:] * 100
    thresholdingVector = hp.findOptimalThresholds(trainingData, 5)
    financeData = np.asarray(pd.DataFrame(data=originalsignal)[trainingLength:])
    testdata = np.asarray(pd.DataFrame(data=originalsignal)[trainingLength:].pct_change())[0:] * 100
    testDataOriginal = np.asarray(originalsignal[trainingLength:].pct_change())[0:] * 100
    print("testData",testdata)
    print("financeData",financeData)
    print("testDataOriginal",testDataOriginal)
    print("originalFinanceSignal",np.asarray(originalsignal[trainingLength:]))

    # trainingdata = trainingdata.squeeze(axis=1)

    """
        
    """
    # s = pd.Series(hp.readDataFromCSV("AMD.CSV")[9000:9400])
    # s = data[1000:3000]
    # testdata = testdata.squeeze(axis=1)
    # Setting evaluater thresholding vector table so that class loss error will be calculated by the evaluater
    model1 = run_agent('LinearRegFor3Day', base=LinearRegAgent)
    lstm_predictor100 = run_agent('lstm_predictor100', base=LSTM_PREDICTOR)
    mlpagentsp = run_agent('mlpAgenNw', base=Mlpagentsp)
    evaluate_agent = run_agent('Evaluator', base=EvaluatorAgent)
    #cnn_agent = run_agent('CNN_Agent', base=CNN_PREDICTOR)
    copyYesterdayAgent = run_agent('CopyYesterdayAgent', base=CopyYesterdayAgent)
    lstm_agent = run_agent('LSTM_Agent', base=LSTM_PREDICTOR)
    lstm_decider = run_agent('LSTM_DECIDER', base=LSTM_DECIDER)
    imitator = run_agent('Imitator', base=ImitatorAgent)
    majorityDecider = run_agent('MajorityDecider', base=MajorityDecider)
    mlpDecider = run_agent('MLPDecider', base=MLPDecider)
    arimaAgent = run_agent('ARIMAAgent', base=ARIMAAgent)
    mlpAgent = run_agent('MLP_AGENT', base=MLPAgent)
    rsiAgent = run_agent('RSIAgent', base=RSIAgent)
    model1.uniqueId = "LinearRegFor3Day"
    lstm_predictor100.uniqueId = "lstm_predictor100"
    mlpagentsp.uniqueId = "mlpagentsp"
    mlpAgent.uniqueId = "mlpAgent"
    mlpDecider.uniqueId = "mlpDecider"
    copyYesterdayAgent.uniqueId = "copyYesterdayAgent"
    # cnn_agent.uniqueId = "cnn_agent"
    arimaAgent.uniqueId = "arima"
    rsiAgent.uniqueId = "rsiAgent"
    evaluate_agent.uniqueId = "evaluater"
    imitator.uniqueId = "imitator"
    lstm_agent.uniqueId = "lstm_agent"
    majorityDecider.uniqueId = "majorityDecider"
    lstm_decider.uniqueId = "lstm_decider"

    # cnn_agent.on_init_properity(3,thresholdingVector)
    mlpagentsp.on_init_properity(40, thresholdingVector)
    mlpAgent.on_init_properity(15, thresholdingVector)
    # copyYesterdayAgent.on_init_properity(thresholdingVector)
    model1.on_init_properity(3, thresholdingVector)
    lstm_predictor100.on_init_properity(100, thresholdingVector)
    lstm_agent.on_init_properity(5, thresholdingVector)
    arimaAgent.on_init_properity(5, thresholdingVector)
    evaluate_agent.on_init_properity(thresholdingVector)
    lstm_decider.on_init_properity(3, thresholdingVector)
    mlpDecider.on_init_properity(3, thresholdingVector)
    # cnn_agent.train(trainingdata)
    mlpagentsp.train(trainingData)
    lstm_agent.train(trainingData)

    mlpAgent.train(trainingData)
    modelsList.append(model1)
    modelsList.append(lstm_agent)
    modelsList.append(arimaAgent)

    modelsList.append(rsiAgent)
    modelsList.append(mlpAgent)
    modelsList.append(mlpagentsp)
    # modelsList.append(cnn_agent)
    modelsList.append(evaluate_agent)
    modelsList.append(imitator)
    modelsList.append(mlpDecider)
    modelsList.append(majorityDecider)
    modelsList.append(lstm_decider)

    return financeData, modelsList, testdata,testDataOriginal,np.asarray(originalsignal[trainingLength:])


if __name__ == '__main__':
    modelsList = []
    filePath = "globalsave"
    ns = run_nameserver()
    server = run_agent('Server', base=Server)

    financeData, modelsList, data,testDataOriginal,originalFinanceSignal = initialize_agent()

    hp.initialConnectionsAgent(modelsList, server)
    # Send messages
    m1 = MessageType()
    # modelsList = loadDatas(modelsList,filePath)
    hp.loadDatas(modelsList, filePath)
    agentlist = hp.getAgentList(modelsList)  # all agent names have been stored in this list

    lstmdeciderLog = []
    mlpdeciderLog = []
    majorityVotingLog = []
    print(np.squeeze(financeData, axis=1))

    for i, d in enumerate(data):
        # In the loop for testing some probabilities
        majorityDeciderFeedBack = []
        if i < 0:
            #print("last five data:",data[i-4:i+1])
            #print("sending value:",d)
            #print("the mean of last five values",np.mean(data[:-5]))
            print("i>7 data",np.expand_dims(np.mean(data[i-14:i+1]),axis=0))
            print("i>7 finance data",np.expand_dims(np.mean(financeData[i-14:i+1]),axis=0))
            m1.message = [np.expand_dims(np.mean(data[i-14:i+1]),axis=0), i - 1,np.expand_dims(np.mean(financeData[i-14:i+1]),axis=0), testDataOriginal[i], originalFinanceSignal[i]]
        else:
            print("i<7 data",d)
            print("i<7 financeData",financeData[i])
            m1.message = [d, i - 1, financeData[i],testDataOriginal[i],originalFinanceSignal[i]]

        if i == 0:
            continue

        """
            Server broadcast the message that have been taking lately from database or real dataset  
        """
        server.server_broadcast(m1)
        print(financeData[i])
        print(len(data))
        """
            After server broadcasting the message and all messages had been reached to agents by server,the evaluater will be updated because of given reasons:
                *Before agents will publish to their behaviours,evaluator should calculate their scores and return to these scores as object to server
                *The priority of the calculating scores can be important due to server priority that depended on agents based system 
        """
        modelsList[6].update()
        """
            In this part of the agent.py code main loop after evaluater collects behaviours of the agents ,they decided to broadcast it to imitator.
            Therefore,at the first part of the code might be sended to a empty behaviours to the agents  
        """
        sendingObjectList = {"LinearRegFor3Day": None, "lstm_agent": None, "arima": None,
                             "rsiAgent": None, "mlpAgent": None,"mlpagentsp":None, "evaluater": None, "imitator": m1, "mlpDecider": None,
                             "majorityDecider": None, "lstm_decider": None}
        m1.message = modelsList[6].getLastBehavioursAgents()
        # # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourTruthLast"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        """
            Agent behaviours are calculating in this scope.If agent will be added as finding possible solution of financial trending ,
            the for each loop range will be increased or changed.
            After passing this block,all the agents can explain their behaviours to server. 
        """
        for j in range(0,6,1):
            modelsList[j].evaluate_behaviour()

        """
            All decisions is sending to evaluater before running majoritydecider or any decider agent.
        """
        sendingObjectList = {"LinearRegFor3Day": None, "lstm_agent": None, "arima": None,
                             "rsiAgent": None, "mlpAgent": None,"mlpagentsp":None, "evaluater": m1,
                             "imitator": None, "mlpDecider": None, "majorityDecider": None, "lstm_decider": None}
        for j in range(0,6, 1):
            m1.message = modelsList[j].get_behaviourstate()
            m1.senderId = modelsList[j].uniqueId
            m1.messageType = "behaviourOfAgentNow"
            majorityDeciderFeedBack.append(m1.message)
            hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        """
            Evaluator is sending their datas to majoritydecider to take a greater score than all agents behaviours 
        """
        sendingObjectList = {"LinearRegFor3Day": None, "lstm_agent": None, "arima": None,
                             "rsiAgent": None, "mlpAgent": None,"mlpagentsp":None,"evaluater": None, "imitator": None,
                             "mlpDecider": m1, "majorityDecider": m1, "lstm_decider": m1}
        m1.message = majorityDeciderFeedBack
        m1.senderId = "evaluater"
        print("message", m1.message)
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        # Decider will be runned in this method because of its priority
        # This priority is more important than some priority that is located in this while loop
        modelsList[-1].evaluate_behaviour()
        modelsList[-2].evaluate_behaviour()
        modelsList[-3].evaluate_behaviour()
        """
            After majoritydecider published its decision,it is sending to majority decider by a  discrete from all general agents
            The purpose of  the majority decider algoritms must obtain a greater score than all agents.Hence,evaluater agents will be 
            argumentative. 
        """
        sendingObjectList = {"LinearRegFor3Day": None, "lstm_agent": None, "arima": None,
                             "rsiAgent": None, "mlpAgent": None,"mlpagentsp":None,"evaluater": m1, "imitator": None, "mlpDecider": None,
                             "majorityDecider": None, "lstm_decider": None}

        m1.message = modelsList[-1].get_behaviourstate()
        m1.senderId = modelsList[-1].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        m1.message = modelsList[-2].get_behaviourstate()
        m1.senderId = modelsList[-2].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        m1.message = modelsList[-3].get_behaviourstate()
        m1.senderId = modelsList[-3].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        """
            Imitator object might be considered as a lstm that try to improve all agents performences.
            The differences between imitator and decider agents is that lstm might be more successful than classical decider according to our plan
        """
        sendingObjectList = {"LinearRegFor3Day": None, "lstm_agent": None, "arima": None,
                             "rsiAgent": None, "mlpAgent": None,"mlpagentsp":None,"evaluater": None, "imitator": m1, "mlpDecider": None,
                             "majorityDecider": None, "lstm_decider": None}
        m1.message = modelsList[6].getAgentLastPredictionList()
        # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        print("time:", i)
        print("data:", financeData[i])
        print("Investmenlist:",modelsList[6].get_agent_total_money_list())
        print("GeneralScores:", modelsList[6].getAgentScores())
        print("PeriodicScores:", modelsList[6].getPeriodicScoreTableAgents())

        if modelsList[6].getPeriodicScoreTableAgents() != {}:
            lstmdeciderLog.append(modelsList[6].getPeriodicScoreTableAgents()["lstm_decider"])
            majorityVotingLog.append(modelsList[6].getPeriodicScoreTableAgents()["majorityDecider"])
            mlpdeciderLog.append(modelsList[6].getPeriodicScoreTableAgents()["mlpDecider"])
        # print("PeriodicDatas",modelsList[3].getPeriodicScoreTableAgents())
    # hp.saveDatas(modelsList, filePath)
    plt.plot(lstmdeciderLog, 'r', label='LstmDecider')  # plotting t, a separately
    plt.plot(majorityVotingLog, 'b', label='majorityVoting')  # plotting t, b separately
    plt.plot(mlpdeciderLog, 'y', label='mlpDecider')  # plotting t, b separately
    plt.legend()
    plt.show()

    # modelsList[5].saveDataFrameCSV()
    ns.shutdown()
