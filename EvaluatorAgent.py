from Model import Model
from MessageType import BehaviourState
import numpy as np
class EvaluatorAgent(Model):
    agentPredictionList = {}
    diffRealBehaviourValue = []
    overallscoreAgents = {}
    overallScoresTableAgents = {}
    periodicScoreTableAgents = {}
    periodOfData = 15

    def receive_agent_message(self,recevingObjectFromAgent):
        if self.agentPredictionList.__contains__(recevingObjectFromAgent.senderId):
            self.agentPredictionList[recevingObjectFromAgent.senderId].append(recevingObjectFromAgent.message)
        else:
            self.agentPredictionList[recevingObjectFromAgent.senderId] = [recevingObjectFromAgent.message]
        return None
    def getScores(self):
        for key in self.agentPredictionList:
            tempPredictionList = np.asarray(self.agentPredictionList[key])
            realList = np.asarray(self.diffRealBehaviourValue)
            self.overallscoreAgents[key] = np.sum(tempPredictionList * realList >= 0) / len(realList)
        return self.overallscoreAgents
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

    def overallScoresTableAgents(self):
        return self.overallScoresTableAgents
    def getPeriodicScoreTableAgents(self):
        return self.periodicScoreTableAgents
    def update(self):
        lenghtMemory = len(self.dataMemory)
        #To take difference between real datas in the real time
        if len(self.diffRealBehaviourValue) != lenghtMemory - 1:
            if self.dataMemory[lenghtMemory - 1] - self.dataMemory[lenghtMemory - 2] > 0:
                 self.diffRealBehaviourValue.append(BehaviourState.BUY)
            if self.dataMemory[lenghtMemory - 1] - self.dataMemory[lenghtMemory - 2] <= 0:
                self.diffRealBehaviourValue.append(BehaviourState.SELL)
        #self.getScores()
        self.calcPeriodicScoresAgents()
        #print("overallScores:",self.overallscoreAgents)
        #print("lastPeriodScores:",self.periodicScoreTableAgents)

    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message)
        self.dataMemory.append(receivingObjectFromServer.message)