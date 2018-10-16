from Model import Model
from MessageType import BehaviourState
import numpy as np
import time
import dill
#Evaluator agent  receives data from other agents that try to  predict the next value of data that has a financial problem or any classification problem
class EvaluatorAgent(Model):
    agentPredictionList = {}
    diffRealBehaviourValue = []
    overallscoreAgents = {}
    periodicScoreTableAgents = {}
    periodOfData = 15
    #When evaluater receive a message from any agent it run receive_agent function
    def receive_agent_message(self,recevingObjectFromAgent):
        if self.agentPredictionList.__contains__(recevingObjectFromAgent.senderId):
            self.agentPredictionList[recevingObjectFromAgent.senderId].append(recevingObjectFromAgent.message)
        else:
            self.agentPredictionList[recevingObjectFromAgent.senderId] = [recevingObjectFromAgent.message]
        return None
        # The method provide to send to message from self to another agent

    def loadALLVariables(self, pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.agentPredictionList = data['agentPredictionList'].tolist()
        self.diffRealBehaviourValue = data['diffRealBehaviourValue'].tolist()
        self.overallscoreAgents = data['overallscoreAgents'].tolist()
        self.periodicScoreTableAgents = data['periodicScoreTableAgents'].tolist()
        self.dataTime = data['dataTime'].tolist()
        self.dataMemory = data['dataMemory'].tolist()


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
        for key in self.agentPredictionList:
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = np.asarray(self.diffRealBehaviourValue)
            self.overallscoreAgents[key] = np.sum(tempPredictionList * realList >= 0) / len(realList)
    #This score give a frame succeed rate.It is different from updateScores due to it calculate truth table of a period of data
    #And it return a table that filled with true or not
    def calcPeriodicScoresAgents(self):
        if len(self.dataMemory) >= self.periodOfData:
            for key in self.agentPredictionList:
                tempPredictionList = np.asarray(self.agentPredictionList[key])
                realList = np.asarray(self.diffRealBehaviourValue)
                lengthPredictList = len(tempPredictionList)
                lengthRealList = len(realList)
                #print(tempPredictionList[lengthPredictList - self.periodOfData:lengthPredictList])
                agentUpdatePredictionList = (tempPredictionList[lengthPredictList - self.periodOfData:lengthPredictList] *
                       realList[lengthRealList - self.periodOfData:lengthRealList] >= 0)
                self.periodicScoreTableAgents[key] = agentUpdatePredictionList
    def getAgentScores(self):
        return self.overallscoreAgents
    def getPeriodicScoreTableAgents(self):
        return self.periodicScoreTableAgents
    #to update flowing of evaluate agent
    #In addition,this update code is including also the piece of code to calculate real behaviour of data
    #After real behaviours of data are calculated,at difference between real value and prediction  is calculated
    def update(self):
        lenghtMemory = len(self.dataMemory)
        #To take difference between real datas in the real time
        if len(self.diffRealBehaviourValue) != lenghtMemory - 1:
            if self.dataMemory[lenghtMemory - 1] - self.dataMemory[lenghtMemory - 2] > 0:
                 self.diffRealBehaviourValue.append(BehaviourState.BUY)
            if self.dataMemory[lenghtMemory - 1] - self.dataMemory[lenghtMemory - 2] <= 0:
                self.diffRealBehaviourValue.append(BehaviourState.SELL)
        self.calcPeriodicScoresAgents()
        self.updateScores()
        #print("overallScores:",self.overallscoreAgents)
        #print("lastPeriodScores:",self.periodicScoreTableAgents)
    #this method recive broadcasting data from server after every broadcast is actualized
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message[0])
        self.dataMemory.append(receivingObjectFromServer.message[0])
        self.dataTime.append(receivingObjectFromServer.message[1])