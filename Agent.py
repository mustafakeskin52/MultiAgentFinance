import time
import numpy as np
import pandas as pd
from Server import Server
from LinearRegAgent import LinearRegAgent
from EvaluatorAgent import EvaluatorAgent
from ImitatorAgent import ImitatorAgent
from MessageType import MessageType
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
from sklearn import linear_model


def initialConnectionsAgent(modelsList):
    # if agent receive a message from server,code will flow to the connectionFunction.
    # Therefore the parameter must been given truthly to could run the code

    for i in range(len(modelsList)):
        modelsList[i].on_init_agent(server, 'receive_server_broadcast_message')

    # to use this function for the purpose of connecting a model to another model
    # one agent is being a listener and another agent is being a publisher
    # model1.connect_to_new_agent(model2,'receive_message')
    # model2.connect(model2.addr('main'),handler = 'receive_message')

    for i in range(len(modelsList)):
        for j in range(len(modelsList)):
            if i != j:
                modelsList[i].connect_to_new_agent(modelsList[j], 'receive_agent_message')
def communicateALLAgents(modelsList,sendingAgent,sendingObjectLists):
    modelIndex = 0
    for index in range(len(modelsList)):
        if modelsList[index].uniqueId == sendingAgent:
            modelIndex = index;
            break;
    for key in sendingObjectLists:
        if sendingAgent != key:
            if  sendingObjectLists[key] != None:
                modelsList[modelIndex].sending_message(sendingObjectLists[key])
            else:
                modelsList[modelIndex].sending_message(None)
def getAgentList(modelLists):
    agentLists = []
    for i in range(len(modelsList)):
        agentLists.append(modelsList[i].uniqueId)
    return agentLists

def evaluateAgentsBehaviours():
    return None
def readDataFromCSV(path):
   spy = pd.read_csv(path)
   data = spy['Adj Close'].values.astype(float)
   return data
if __name__ == '__main__':
    modelsList = []
    ns = run_nameserver()
    model1 = run_agent('Model1', base=LinearRegAgent)
    model2 = run_agent('Model2', base=LinearRegAgent)
    model3 = run_agent('Model3',base=LinearRegAgent)
    evaluate_agent = run_agent('Evaluator',base=EvaluatorAgent)
    imitator = run_agent('Imitator',base=ImitatorAgent)
    server = run_agent('Server', base=Server)

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

    initialConnectionsAgent(modelsList)
    # Send messages
    m1 = MessageType()

    agentlist = getAgentList(modelsList)#all agent names have been stored in this list
    data = readDataFromCSV("AMD.CSV")
    for i in range(len(data)):
        #In the loop for testing some probabilities
        m1.message = [data[i],i]
        server.server_broadcast(m1)

        modelsList[3].update()
        print("realIncreasing", data[i] - data[i - 1])

        sendingObjectList = {"model1": None, "model2": None, "model3": None, "evaluater": None, "imitator": m1}
        m1.message = modelsList[3].getPeriodicScoreTableAgents()
        # print("list",m1.message)
        m1.senderId = "evaluater"
        m1.messageType = "behaviourOfAgentNow"

        communicateALLAgents(modelsList, m1.senderId, sendingObjectList)
        print(modelsList[4].getbehaviourTruthTableNow())

        modelsList[0].evaluate_behaviour(3)
        modelsList[1].evaluate_behaviour(5)
        modelsList[2].evaluate_behaviour(7)

        sendingObjectList = {"model1": None, "model2": None, "model3": None,"evaluater":m1,"imitator":None}

        for j in range(0,3,1):
            m1.message = modelsList[j].get_behaviourstate()
            m1.senderId = modelsList[j].uniqueId
            m1.messageType = "behaviourOfAgentNow"
            communicateALLAgents(modelsList,m1.senderId,sendingObjectList)

        print("model1:",modelsList[0].get_behaviourstate())
        print("model2:",modelsList[1].get_behaviourstate())
        print("model3:", modelsList[2].get_behaviourstate())
        print(modelsList[3].getAgentScores())
    ns.shutdown()