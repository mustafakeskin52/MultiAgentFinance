import time
import numpy as np
import os.path
import pandas as pd
import pickle
import Agent as ag
from Server import Server
from LinearRegAgent import LinearRegAgent
from EvaluatorAgent import EvaluatorAgent
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
    model3 = run_agent('Model3',base=LinearRegAgent)
    evaluate_agent = run_agent('Evaluator',base=EvaluatorAgent)
    imitator = run_agent('Imitator',base=ImitatorAgent)

    model1.uniqueId = "model1"
    model2.uniqueId = "model2"
    model3.uniqueId = "model3"
    evaluate_agent.uniqueId = "evaluater"
    imitator.uniqueId = "imitator"

    modelsList.append(model1)
    modelsList.append(model2)
    modelsList.append(model3)
    modelsList.append(evaluate_agent)
    modelsList.append(imitator)

    ag.initialConnectionsAgent(modelsList)
    # Send messages
    m1 = MessageType()
    #modelsList = loadDatas(modelsList,filePath)
    ag.loadDatas(modelsList,filePath)
    agentlist = ag.getAgentList(modelsList)#all agent names have been stored in this list
    data = ag.readDataFromCSV("AMD.CSV")
    for i in range(1800,len(data),1):
        #In the loop for testing some probabilities
        m1.message = [data[i], i]
        server.server_broadcast(m1)

        modelsList[3].update()
        print("realIncreasing", data[i] - data[i - 1])

        sendingObjectList = {"model1": None, "model2": None, "model3": None, "evaluater": None, "imitator": m1}
        m1.message = modelsList[3].getPeriodicScoreTableAgents()
        # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourOfAgentNow"

        ag.communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        #print(modelsList[4].getbehaviourTruthTableNow())

        modelsList[0].evaluate_behaviour(3)
        modelsList[1].evaluate_behaviour(5)
        modelsList[2].evaluate_behaviour(7)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"evaluater":m1,"imitator":None}

        for j in range(0,3,1):
            m1.message = modelsList[j].get_behaviourstate()
            m1.senderId = modelsList[j].uniqueId
            m1.messageType = "behaviourOfAgentNow"
            ag.communicateALLAgents(modelsList,m1.senderId,sendingObjectList)
        if i % 100 == 0:
            ag.saveDatas(modelsList, filePath)
        #print("model1:",modelsList[0].get_behaviourstate())
        #print("model2:",modelsList[1].get_behaviourstate())
        #print("model3:", modelsList[2].get_behaviourstate())
        print(modelsList[3].getAgentScores())

    ns.shutdown()
