from ModelAgent import Model
from MessageType import BehaviourState
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import dill
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
    agentEvaluationStartData = 0
    diffRealBehaviourValue = []
    overallscoreAgents = {}
    periodicScoreTableAgents = {}
    periodOfData = 10
    scoreOfTheLastBehaviours = {}
    thresholdArray = []

    def on_init_properity(self,thresholding):
        self.thresholdArray = thresholding
    # When evaluater receive a message from any agent it run receive_agent function

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
        for key in self.agentPredictionList:
            print("name:",key)
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = self.dataClassMemory[1:]
            # confusionmatrix = confusion_matrix(realList, tempPredictionList, labels=[4, 3, 2,1,0])
            # np.set_printoptions(precision=2)
            # #To normalize row of the confusion matrix
            # drawConfusionMatrix = []
            # for i, d in enumerate(np.sum(confusionmatrix,axis=1)):
            #     drawConfusionMatrix.append(confusionmatrix[i, :] / d)
            #
            # print(np.asarray(drawConfusionMatrix))
            # print(np.sum(confusionmatrix,axis=1))
            self.overallscoreAgents[key] = np.sum(tempPredictionList == realList) / len(realList)

        # This score give a frame succeed rate.It is different from updateScores due to it calculate truth table of a period of data
        # And it return a table that filled with true or not

    def calcLastBehavioursAgents(self):
        for key in self.agentPredictionList:
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = self.dataClassMemory[1:]
            agentsScoreList = int(tempPredictionList[len(tempPredictionList) - 1] == realList[len(realList) - 1])
            self.scoreOfTheLastBehaviours[key] = agentsScoreList
    #This score give a frame succeed rate.It is different from updateScores due to it calculate truth table of a period of data
    #And it return a table that filled with true or not
    def calcPeriodicScoresAgents(self):
        for key in self.agentPredictionList:
            print("name:", key)
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = self.dataClassMemory[1:]
            confusionmatrix = confusion_matrix(realList[-self.periodOfData:], tempPredictionList[-self.periodOfData:], labels=[4, 3, 2, 1, 0])
            np.set_printoptions(precision=2)
            # To normalize row of the confusion matrix
            drawConfusionMatrix = []
            for i, d in enumerate(np.sum(confusionmatrix, axis=1)):
                drawConfusionMatrix.append(confusionmatrix[i, :] / d)

            print(np.asarray(drawConfusionMatrix))
            print(np.sum(confusionmatrix, axis=1))
            tempScores = np.sum(tempPredictionList[-self.periodOfData:] == realList[-self.periodOfData:])/len(realList[-self.periodOfData:])
            self.periodicScoreTableAgents[key] = tempScores
    def getAgentPredictions(self):
        return self.agentPredictionList
    def getAgentScores(self):
        return self.overallscoreAgents
    def getLastBehavioursAgents(self):
        return self.scoreOfTheLastBehaviours
    def getPeriodicScoreTableAgents(self):
        return self.periodicScoreTableAgents
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
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message[0])
        temp = 0
        if receivingObjectFromServer.message[0] > self.thresholdArray[0]:
            temp = BehaviourState.HIGH_BUY
        elif receivingObjectFromServer.message[0] > self.thresholdArray[1]:
            temp = BehaviourState.BUY
        elif receivingObjectFromServer.message[0] > self.thresholdArray[2]:
            temp = BehaviourState.NONE
        elif receivingObjectFromServer.message[0] > self.thresholdArray[3]:
            temp = BehaviourState.SELL
        else:
            temp = BehaviourState.LOW_SELL
        self.dataTime.append(receivingObjectFromServer.message[1])
        self.dataClassMemory.append(temp)