from ModelAgent import Model
from MessageType import BehaviourState
import numpy as np
from sklearn.metrics import confusion_matrix

a = np.asarray([[6,6,6],[5,5,5],[4,4,4]])
print( )
print(a)
for i,d in enumerate(np.sum(a,axis=1)):
    print(a[i, :]/d)