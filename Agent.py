import time
import os.path
import pickle
import HelperFunctions as hp
import numpy as np
from Server import Server
from ARIMAAgent import ARIMAAgent
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

if __name__ == '__main__':
    modelsList = []
    filePath = "globalsave"
    ns = run_nameserver()

    server = run_agent('Server', base=Server)
    model1 = run_agent('Model1', base=LinearRegAgent)
    model2 = run_agent('Model2', base=LinearRegAgent)
    model3 = run_agent('Model3',base=MovingAverageAgent)
    evaluate_agent = run_agent('Evaluator',base=EvaluatorAgent)
    imitator = run_agent('Imitator',base=ImitatorAgent)
    majorityDecider = run_agent('MajorityDecider',base=MajorityDecider)
    arimaAgent = run_agent('ARIMAAgent',base=ARIMAAgent)

    model1.uniqueId = "model1"
    model2.uniqueId = "model2"
    model3.uniqueId = "model3"
    arimaAgent.uniqueId = "arima"
    evaluate_agent.uniqueId = "evaluater"
    imitator.uniqueId = "imitator"
    majorityDecider.uniqueId = "majorityDecider"

    modelsList.append(model1)
    modelsList.append(model2)
    modelsList.append(model3)
    modelsList.append(arimaAgent)
    modelsList.append(evaluate_agent)
    modelsList.append(imitator)
    modelsList.append(majorityDecider)

    hp.initialConnectionsAgent(modelsList,server)
    # Send messages
    m1 = MessageType()

    #modelsList = loadDatas(modelsList,filePath)
    hp.loadDatas(modelsList,filePath)
    agentlist = hp.getAgentList(modelsList)#all agent names have been stored in this list
    #data =hp.sinData(1000,30)# np.add(hp.sinData(1000,30), hp.sinData(1000,50))#hp.sinData(1000,30)#np.add(hp.sinData(1000,30), hp.sinData(1000,50))
    data = hp.readDataFromCSV("AMD.CSV")[7000:]
    for i,d in enumerate(data):
        #In the loop for testing some probabilities
        majorityDeciderFeedBack = []
        m1.message = [d, i]
        server.server_broadcast(m1)

        print(len(data))
        modelsList[4].update()
        print("time:",i)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"arima":None,"evaluater": None, "imitator": m1,"majorityDecider":None}
        m1.message = modelsList[4].getLastBehavioursAgents()
        # # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourTruthLast"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        modelsList[0].evaluate_behaviour(3)
        modelsList[1].evaluate_behaviour(5)
        modelsList[2].evaluate_behaviour(3)
        modelsList[3].evaluate_behaviour(5)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"arima":None,"evaluater":m1,"imitator":None,"majorityDecider":None}

        for j in range(0,4,1):
            m1.message = modelsList[j].get_behaviourstate()
            m1.senderId = modelsList[j].uniqueId
            m1.messageType = "behaviourOfAgentNow"
            majorityDeciderFeedBack.append(m1.message)
            hp.communicateALLAgents(modelsList,m1.senderId,sendingObjectList)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"arima":None,"evaluater": None, "imitator": None,
                             "majorityDecider": m1}
        m1.message = majorityDeciderFeedBack
        m1.senderId = "evaluater"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        #Decider will be runned in this method because of its priority
        #This priority is more important than some priority that is located in this while loop
        modelsList[-1].evaluate_behaviour()

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"arima":None, "evaluater": m1, "imitator": None,
                             "majorityDecider": None}

        m1.message = modelsList[-1].get_behaviourstate()
        m1.senderId = modelsList[-1].uniqueId
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"arima":None,"evaluater": None, "imitator": m1,"majorityDecider": None}
        m1.message = modelsList[4].getAgentLastPredictionList()
        # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourOfAgentNow"
        hp.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)

        print("GeneralScores:",modelsList[4].getAgentScores())
        print("PeriodicScores:",modelsList[4].getPeriodicScoreTableAgents())
        #print("PeriodicDatas",modelsList[3].getPeriodicScoreTableAgents())
    #hp.saveDatas(modelsList, filePath)
    modelsList[5].saveDataFrameCSV()
    ns.shutdown()
