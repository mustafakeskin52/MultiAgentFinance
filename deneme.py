from tempfile import TemporaryFile
import numpy as np

outfile = TemporaryFile()
x = np.arange(10)
y = np.arange(10)
np.save("trainingXVariables",x)
np.save("trainingYVariables",y)
x = np.load("trainingXVariables.npy")
y = np.load("trainingYVariables.npy")
#print(y)
