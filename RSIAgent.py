from ModelAgent import Model
import numpy as np
from sklearn import linear_model

class BehaviourState:
    HIGH_BUY = 4
    BUY = 3
    NONE = 2
    SELL = 1
    LOW_SELL = 0

#A model might extend to class that is a abstract agent model including basic layouts
class RSIAgent(Model):

    def receive_agent_message(self,receivingObjectFromAgent):
        if receivingObjectFromAgent != None:
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.senderId)
            self.log_info('ReceivedFromAgent: %s' % receivingObjectFromAgent.message)

    def loadALLVariables(self, pathOfImitatorObject):
        data = np.load(pathOfImitatorObject)
        self.dataMemory = data['dataMemory'].tolist()
        self.dataTime = data['dataTime'].tolist()

    def saveALLVariables(self, pathOfImitatorObject):
        np.savez(pathOfImitatorObject,dataMemory=self.dataMemory,
                 dataTime=self.dataTime)
    # The method provide to send to message from self to another agent
    def evaluate_behaviour(self):
        periodLength = 15
        signal = np.squeeze(np.asarray(self.signalMemory),axis = 1)
        if len(signal) >= periodLength:
            #print(t)
            print("signal:",signal[-periodLength:])
            diff = np.diff(signal[-periodLength:])

            print("diff:",diff)
            upValues = [i for i in diff if i >= 0]

            print("upValues:",upValues)
            downValues = [i for i in diff if i < 0]
            downValues = np.abs(downValues)

            print("downValues:",downValues)
            AvgU = np.sum(upValues)/periodLength
            AvgD = np.sum(downValues)/periodLength

            Rs = AvgU/AvgD
            RSI = 100 - 100/(1 + Rs)
            print("RSI",RSI)
            if RSI > 65:
                self.behaviourState = BehaviourState.HIGH_BUY
            elif RSI > 55:
                self.behaviourState = BehaviourState.BUY
            elif RSI > 45:
                self.behaviourState = BehaviourState.NONE
            elif RSI > 35:
                self.behaviourState = BehaviourState.SELL
            else:
                self.behaviourState = BehaviourState.LOW_SELL