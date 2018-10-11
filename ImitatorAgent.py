from Model import Model
import numpy as np
class ImitatorAgent(Model):
    overallScoreAgents = {}
    behaviourTruthTableNow = {}
    behaviourTruthTableAll = {}
    framePeriodOfData = 30

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
    def receive_server_broadcast_message(self, receivingObjectFromServer):
        self.log_info('ReceivedFromServer: %s' % receivingObjectFromServer.message)
        self.dataMemory.append(receivingObjectFromServer.message)
    def getoverallScoreAgents(self):
        return self.overallScoreAgents
    def getbehaviourTruthTableNow(self):
        return self.behaviourTruthTableNow
    def getbehaviourOfAgentsAll(self):
        return self.behaviourTruthTableAll