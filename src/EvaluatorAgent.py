from ModelAgent import Model
from MessageType import BehaviourState
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
import dill
from tempfile import TemporaryFile

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

#Evaluator agent  receives data from other agents that try to  predict the next value of data that has a financial problem or any classification problem
class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0
class EvaluatorAgent(Model):

    agentLastPredictionList = []
    agentPredictionList = {}
    agent_total_money_list = {}
    agentEvaluationStartData = 0
    diffRealBehaviourValue = []
    overallscoreAgents = {}
    periodicScoreTableAgents = {}
    periodOfData = 500
    scoreOfTheLastBehaviours = {}
    thresholdArray = []
    startPointScoresCalc = 200

    def on_init_properity(self,thresholding):
        self.thresholdArray = thresholding
    # When evaluater receive a message from any agent it run receive_agent function

    def receive_agent_message(self,recevingObjectFromAgent):

        if len(self.dataClassMemory) > self.startPointScoresCalc:
            if self.agentPredictionList.__contains__(recevingObjectFromAgent.senderId):
                self.agentPredictionList[recevingObjectFromAgent.senderId].append(recevingObjectFromAgent.message)
                tempvar = self.agent_total_money_list[recevingObjectFromAgent.senderId]
                money,amountinvestment = self.res_agent_behaviour(recevingObjectFromAgent.message,tempvar[1],tempvar[0])
                self.agent_total_money_list[recevingObjectFromAgent.senderId] = [money,amountinvestment]
            else:
                self.agentPredictionList[recevingObjectFromAgent.senderId] = [recevingObjectFromAgent.message]
                self.agent_total_money_list[recevingObjectFromAgent.senderId] = [1000.0,0]
            return None
        # The method provide to send to message from self to another agent
    def res_agent_behaviour(self,behaviour,amountinvestment,money):
        signal_memory = np.squeeze(self.signalMemory)

        if behaviour == BehaviourState.HIGH_BUY and amountinvestment == 0:
            amountinvestment = money /signal_memory[-1]
            money = 0
        if  behaviour == BehaviourState.BUY and amountinvestment == 0:
            amountinvestment = money /signal_memory[-1]
            money = 0
        if behaviour == BehaviourState.LOW_SELL and amountinvestment != 0:
            money = amountinvestment*signal_memory[-1]
            amountinvestment = 0
        if behaviour == BehaviourState.SELL and amountinvestment != 0:
            money = amountinvestment * signal_memory[-1]
            amountinvestment = 0

        return money,amountinvestment

    def loadALLVariables(self, pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.agentPredictionList = data['agentPredictionList'].tolist()
        self.diffRealBehaviourValue = data['diffRealBehaviourValue'].tolist()
        self.overallscoreAgents = data['overallscoreAgents'].tolist()
        self.periodicScoreTableAgents = data['periodicScoreTableAgents'].tolist()
        self.dataTime = data['dataTime'].tolist()
        self.dataMemory = data['dataMemory'].tolist()
    def getAgentLastPredictionList(self):
        self.agentLastPredictionList = []
        for key in self.agentPredictionList:
            self.agentLastPredictionList.append(self.agentPredictionList[key][:])
        return self.agentLastPredictionList
    def saveALLVariables(self, pathOfImitatorObject):
        np.savez(pathOfImitatorObject, agentPredictionList=self.agentPredictionList,
                diffRealBehaviourValue=self.diffRealBehaviourValue,
                overallscoreAgents=self.overallscoreAgents,
                periodicScoreTableAgents=self.periodicScoreTableAgents,
                dataMemory =self.dataMemory,
                dataTime=self.dataTime)
    #This score show that all of agents succeed to predict next value of datas rightly.
    #This score is a rate over 1.00
    def updateScores(self):
        realList = []
        data_length = 0;
        for key in self.agentPredictionList:

            print("Evaluation of the :",key)
            tempPredictionList = np.squeeze(np.asarray(self.agentPredictionList[key]))
            realList = np.squeeze(np.asarray(self.dataClassMemory[-tempPredictionList.size:]))

            #if data_length % 50 == 0:
             #   plt.plot(tempPredictionList)
            print("Loss:",np.mean(np.square(tempPredictionList - realList)))
            print("Score",np.count_nonzero(tempPredictionList*realList>0)/tempPredictionList.size)
            np.save(key, np.squeeze(np.asarray(self.agentPredictionList[key])))

            print("Corelation between lstm decider and ", key)
            print(np.corrcoef(tempPredictionList,np.squeeze(np.asarray(self.agentPredictionList["lstm_decider"]))))
            distance, path = fastdtw(np.expand_dims(np.squeeze(np.asarray(self.agentPredictionList["lstm_decider"])),axis=0), np.expand_dims(tempPredictionList,axis=0), dist=euclidean)
            print("Prediction distance between lstm decider and",key)
            print(distance)


       # plt.plot(np.random.rand(30))  # plotting t, b separately
       # plt.legend()
       # plt.show()

    def calcLastBehavioursAgents(self):
        for key in self.agentPredictionList:
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = self.dataClassMemory[-len(tempPredictionList):]
            agentsScoreList = int(tempPredictionList[len(tempPredictionList) - 1] == realList[len(realList) - 1])
            self.scoreOfTheLastBehaviours[key] = agentsScoreList
    #This score give a frame succeed rate.It is different from updateScores due to it calculate truth table of a period of data
    #And it return a table that filled with true or not
    def calcPeriodicScoresAgents(self):
        for key in self.agentPredictionList:
            print("name:", key)
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = self.dataClassMemory[-len(tempPredictionList):]
            # confusionmatrix = confusion_matrix(realList[-self.periodOfData:], tempPredictionList[-self.periodOfData:], labels=[4, 3, 2, 1, 0])
            # np.set_printoptions(precision=2)
            # # To normalize row of the confusion matrix
            # drawConfusionMatrix = []
            # for i, d in enumerate(np.sum(confusionmatrix, axis=1)):
            #     drawConfusionMatrix.append(confusionmatrix[i, :] / d)
            #
            # print(np.asarray(drawConfusionMatrix))
            # print(np.sum(confusionmatrix, axis=1))
            tempScores = np.sum(tempPredictionList[-self.periodOfData:] == realList[-self.periodOfData:])/len(realList[-self.periodOfData:])
            self.periodicScoreTableAgents[key] = tempScores
    def get_agent_total_money_list(self):
        return self.agent_total_money_list
    def getAgentPredictions(self):
        return self.agentPredictionList
    def getAgentScores(self):
        return self.overallscoreAgents
    def getLastBehavioursAgents(self):
        return self.scoreOfTheLastBehaviours
    def getPeriodicScoreTableAgents(self):
        return self.periodicScoreTableAgents
    def getreallist(self):
        tempPredictionList = np.squeeze(np.asarray(self.agentPredictionList["lstm_decider"]))
        return np.squeeze(np.asarray(self.dataClassMemory[-tempPredictionList.size:]))
    #to update flowing of evaluater agent
    #In addition,this update code is including also the piece of code to calculate real behaviour of data
    #After real behaviours of data are calculated,at difference between real value and prediction  is calculated
    def update(self):
        lenghtMemory = len(self.dataMemory)

        self.calcPeriodicScoresAgents()
        self.updateScores()
        self.calcLastBehavioursAgents()
        #print("overallScores:",self.overallscoreAgents)
        #print("lastPeriodScores:",self.periodicScoreTableAgents)
    #this method recive broadcasting data from server after every broadcast is actualized
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServerEvaluater: %s' % receivingObjectFromServer.message[3])
        temp = 0
        if receivingObjectFromServer.message[3] != None:
            if receivingObjectFromServer.message[3] > self.thresholdArray[0]:
                temp = BehaviourState.HIGH_BUY
            elif receivingObjectFromServer.message[3] > self.thresholdArray[1]:
                temp = BehaviourState.BUY
            elif receivingObjectFromServer.message[3] > self.thresholdArray[2]:
                temp = BehaviourState.NONE
            elif receivingObjectFromServer.message[3] > self.thresholdArray[3]:
                temp = BehaviourState.SELL
            else:

                temp = BehaviourState.LOW_SELL
        self.dataTime.append(receivingObjectFromServer.message[1])
        self.dataClassMemory.append(receivingObjectFromServer.message[3])
        self.signalMemory.append(receivingObjectFromServer.message[4])