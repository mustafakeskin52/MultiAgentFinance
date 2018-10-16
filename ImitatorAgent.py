from Model import Model
import numpy as np
import dill
import time
class ImitatorAgent(Model):
    overallScoreAgents = {}
    behaviourTruthTableNow = {}
    behaviourTruthTableAll = {}
    lstmInputSetLastPeriod = []
    lstmOutputSetY = []
    lstmOutputSetX = []
    lstmOutputSetNumpyX = []
    lstmOutputSetNumpyY = []
    periodOfData = 15
    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            if receivingObjectFromAgent.messageType == "overAllScores":
                self.overallScoreAgents = receivingObjectFromAgent.message
            if receivingObjectFromAgent.messageType == "behaviourOfAgentNow":
                self.behaviourTruthTableNow = receivingObjectFromAgent.message
                self.updateBehaviourAllTable()
        # The method provide to send to message from self to another agent

    def loadALLVariables(self,pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.overallScoreAgents = data['overallScoreAgents'].tolist()
        self.behaviourTruthTableNow= data['behaviourTruthTableNow'].tolist()
        self.behaviourTruthTableAll= data['behaviourTruthTableAll'].tolist()
        self.lstmInputSetLastPeriod= data['lstmInputSetLastPeriod']
        self.lstmOutputSetY= data['lstmOutputSetY'].tolist()
        self.lstmOutputSetX= data['lstmOutputSetX'].tolist()
        self.lstmOutputSetNumpyX = data['lstmOutputSetNumpyX']
        self.lstmOutputSetNumpyY = data['lstmOutputSetNumpyY']
        self.dataMemory = data['dataMemory'].tolist()
        self.dataTime = data['dataTime'].tolist()
        return
    def saveALLVariables(self,pathOfImitatorObject):
        np.savez(pathOfImitatorObject,overallScoreAgents =self.overallScoreAgents,
                behaviourTruthTableNow = self.behaviourTruthTableNow,
                behaviourTruthTableAll = self.behaviourTruthTableAll,
                lstmInputSetLastPeriod = self.lstmInputSetLastPeriod,
                lstmOutputSetY = self.lstmOutputSetY,
                lstmOutputSetX = self.lstmOutputSetX,
                lstmOutputSetNumpyX = self.lstmOutputSetNumpyX,
                lstmOutputSetNumpyY = self.lstmOutputSetNumpyY,
                dataMemory = self.dataMemory,
                dataTime = self.dataTime)
    def saveTrainingVariables(self,pathOfXVariables,pathOfYVariables):
        np.save(pathOfXVariables, self.lstmOutputSetNumpyX)
        np.save(pathOfYVariables, self.lstmOutputSetNumpyY)
    def loadTrainingVaribles(self,pathOfXVariables,pathOfYVariables):
        self.lstmOutputSetNumpyX = np.load(pathOfXVariables)
        self.lstmOutputSetNumpyY = np.load(pathOfYVariables)
    def updateBehaviourAllTable(self):
        print("lstmOutputLength",len(self.lstmOutputSetNumpyY))
        for key in self.behaviourTruthTableNow:
            if self.behaviourTruthTableAll.__contains__(key):
                self.behaviourTruthTableAll[key].append(self.behaviourTruthTableNow[key])
            else:
                self.behaviourTruthTableAll[key] = [self.behaviourTruthTableNow[key]]
        self.updateListOfTrain()
        self.listToNumpyArray()
    #update to data before all process is started
    def updateListOfTrain(self):
        if len(self.lstmInputSetLastPeriod) == self.periodOfData:
            self.lstmOutputSetY.append(np.asarray(self.lstmInputSetLastPeriod[-1,:],dtype=int))
            self.lstmOutputSetX.append(np.asarray(self.lstmInputSetLastPeriod[:-1,:],dtype=int))
    #List variables is converted to numpy array variables for giving a lstm network
    def listToNumpyArray(self):
        self.lstmInputSetLastPeriod = list(self.behaviourTruthTableNow.values())
        self.lstmInputSetLastPeriod = np.asarray(self.lstmInputSetLastPeriod ,dtype=int)
        self.lstmInputSetLastPeriod = np.transpose(self.lstmInputSetLastPeriod)

        self.lstmOutputSetNumpyY = np.asarray(self.lstmOutputSetY)
        self.lstmOutputSetNumpyX = np.asarray(self.lstmOutputSetX)

    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.dataMemory.append(receivingObjectFromServer.message[0])
        self.dataTime.append(receivingObjectFromServer.message[1])

    def getoverallScoreAgents(self):
        return self.overallScoreAgents
    def getbehaviourTruthTableNow(self):
        return self.behaviourTruthTableNow
    def getbehaviourOfAgentsAll(self):
        return self.behaviourTruthTableAll