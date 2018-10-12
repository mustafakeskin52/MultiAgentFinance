from Model import Model
import numpy as np
class ImitatorAgent(Model):
    overallScoreAgents = {}
    behaviourTruthTableNow = {}
    behaviourTruthTableAll = {}
    lstmInputSetLastPeriod = []
    lstmOutputSetY = []
    lstmOutputSetX = []
    lstmOutputSetNumpyX = []
    lstmOutputSetNumpyY = []

    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            if receivingObjectFromAgent.messageType == "overAllScores":
                self.overallScoreAgents = receivingObjectFromAgent.message
            if receivingObjectFromAgent.messageType == "behaviourOfAgentNow":
                self.behaviourTruthTableNow = receivingObjectFromAgent.message
                self.updateBehaviourAllTable()

    def updateBehaviourAllTable(self):
        for key in self.behaviourTruthTableNow:
            if self.behaviourTruthTableAll.__contains__(key):
                self.behaviourTruthTableAll[key].append(self.behaviourTruthTableNow[key])
            else:
                self.behaviourTruthTableAll[key] = [self.behaviourTruthTableNow[key]]
        self.updateListOfTrain()
        self.listToNumpyArray()
    #update to data before all process is started
    def updateListOfTrain(self):
        if self.lstmInputSetLastPeriod != []:
            self.lstmOutputSetY.append(self.lstmInputSetLastPeriod[-1,:])
            self.lstmOutputSetX.append(self.lstmInputSetLastPeriod[:-1,:])
    #List variables is converted to numpy array variables for giving a lstm network
    def listToNumpyArray(self):
        self.lstmInputSetLastPeriod = list(self.behaviourTruthTableNow.values())
        self.lstmInputSetLastPeriod = np.array(self.lstmInputSetLastPeriod ,dtype=int)
        self.lstmInputSetLastPeriod = np.transpose(self.lstmInputSetLastPeriod)

        #print(self.lstmOutputSetY)
        self.lstmOutputSetNumpyY = np.asarray(self.lstmOutputSetY)
        self.lstmOutputSetNumpyX = np.asarray(self.lstmOutputSetX)
        print(self.lstmOutputSetNumpyX)
        print(self.lstmOutputSetNumpyY)
        #self.lstmOutputSetNumpyX = np.transpose(self.lstmOutputSetNumpyX)
        #print(self.lstmOutputSetNumpyX)
        #self.lstmOutputSetNumpyY = np.array(self.lstmOutputSetY, dtype=int)
        #self.lstmOutputSetNumpyY = np.transpose(self.lstmOutputSetNumpyX)

    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.dataMemory.append(receivingObjectFromServer.message[0])
        self.dataTime.append(receivingObjectFromServer.message[1])
    def getoverallScoreAgents(self):
        return self.overallScoreAgents
    def getbehaviourTruthTableNow(self):
        return self.behaviourTruthTableNow
    def getbehaviourOfAgentsAll(self):
        return self.behaviourTruthTableAll