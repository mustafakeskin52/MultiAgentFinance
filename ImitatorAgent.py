from ModelAgent import Model
import numpy as np
import pandas as pd
import dill
import time
class ImitatorAgent(Model):
    overallScoreAgents = {}
    behaviourTruthTableLast = {}
    behaviourTruthTableAll = {}
    lstmOutputSetLastPeriod = []
    lstmOutputSetOverall = []
    lstmInputSetOverall = []

    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            if receivingObjectFromAgent.message != {}:
                if receivingObjectFromAgent.messageType == "overAllScores":
                    self.overallScoreAgents = receivingObjectFromAgent.message
                elif receivingObjectFromAgent.messageType == "behaviourTruthLast":
                    self.behaviourTruthTableLast = receivingObjectFromAgent.message
                    self.updateBehaviourAllTable()
                elif receivingObjectFromAgent.messageType == "behaviourOfAgentNow":
                    self.lstmInputSetOverall = receivingObjectFromAgent.message
    # The method provide to send to message from self to another agent
    def loadALLVariables(self,pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.overallScoreAgents = data['overallScoreAgents'].tolist()
        self.behaviourTruthTableLast= data['behaviourTruthTableLast'].tolist()
        self.behaviourTruthTableAll= data['behaviourTruthTableAll'].tolist()
        self.lstmOutputSetLastPeriod= data['lstmOutputSetLastPeriod']
        self.lstmInputSetOverall = data['lstmInputSetOverall']
        self.lstmOutputSetOverall = data['lstmOutputSetOverall'].tolist()
        self.dataMemory = data['dataMemory'].tolist()
        self.dataTime = data['dataTime'].tolist()
    def saveALLVariables(self,pathOfImitatorObject):
        np.savez(pathOfImitatorObject,overallScoreAgents =self.overallScoreAgents,
                  behaviourTruthTableLast = self.behaviourTruthTableLast,
                 behaviourTruthTableAll = self.behaviourTruthTableAll,
                 lstmOutputSetLastPeriod = self.lstmOutputSetLastPeriod,
                 lstmOutputSetOverall = self.lstmOutputSetOverall,
                 lstmInputSetOverall = self.lstmInputSetOverall,
                 dataMemory = self.dataMemory,
                 dataTime = self.dataTime)
    def saveDataFrameCSV(self):
        df = pd.DataFrame(self.lstmInputSetOverall)
        df = df.iloc[:, :-1]
        df1_transposed = df.T
        df1_transposed.to_csv("xtrain.csv", sep=',', encoding='utf-8',index=False)

        templstmOutputSetOverall = np.asarray(self.lstmOutputSetOverall).transpose()
        df2 = pd.DataFrame(templstmOutputSetOverall)
        df2_transposed = df2.T
        df2_transposed.to_csv("ytrain.csv", sep=',', encoding='utf-8',index=False)
    def updateBehaviourAllTable(self):
        self.lstmOutputSetLastPeriod = list(self.behaviourTruthTableLast.values())
        self.lstmOutputSetOverall.append(self.lstmOutputSetLastPeriod)
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.dataMemory.append(receivingObjectFromServer.message[0])
        self.dataTime.append(receivingObjectFromServer.message[1])

    def getoverallScoreAgents(self):
        return self.overallScoreAgents
    def getbehaviourTruthTableNow(self):
        return self.behaviourTruthTableNow
    def getbehaviourOfAgentsAll(self):
        return self.behaviourTruthTableAll