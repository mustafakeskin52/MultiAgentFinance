import time
import numpy as np
import os.path
import pandas as pd
import pickle
from Server import Server
from LinearRegAgent import LinearRegAgent
from EvaluatorAgent import EvaluatorAgent
from ImitatorAgent import ImitatorAgent
from MessageType import MessageType
from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
from sklearn import linear_model

def initialConnectionsAgent(modelsList,server):
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

def getAgentList(modelsLists):
    agentLists = []
    for i in range(len(modelsLists)):
        agentLists.append(modelsLists[i].uniqueId)
    return agentLists

def evaluateAgentsBehaviours():
    return None
def sinData(fs,f):
    x = np.arange(fs)  # the points on the x axis for plotting
    # compute the value (amplitude) of the sin wave at the for each sample
    return [np.sin(2 * np.pi * f * (i / fs)) for i in x]
def cosData(fs,f):
    x = np.arange(fs)  # the points on the x axis for plotting
    # compute the value (amplitude) of the sin wave at the for each sample
    return [np.cos(2 * np.pi * f * (i / fs)) for i in x]
def readDataFromCSV(path):
   spy = pd.read_csv(path)
   data = spy['Adj Close'].values.astype(float)
   return data
def loadDatas(modelsList,fileName):
    for i in range(len(modelsList)):
        if os.path.exists(fileName+str(i)+".npz") == True:
            modelsList[i].loadALLVariables(fileName+str(i)+".npz")
    return modelsList
def saveDatas(modelsList,fileName):
    for i in range(len(modelsList)):
        if os.path.exists(fileName + str(i) + ".npz") == True:
            os.remove(fileName + str(i) + ".npz")
    for t in range(len(modelsList)):
        modelsList[t].saveALLVariables(fileName + str(t) + ".npz")