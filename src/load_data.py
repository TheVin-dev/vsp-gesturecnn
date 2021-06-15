import numpy as np 
import os 
import numpy as np 
import sys


def loadData(label):

    cwd = os.getcwd() 
    parentDir = os.path.dirname(cwd)
    dataDir = os.path.join(parentDir,"data")
    data = None
    files = [f for f in os.listdir(dataDir)]

    if len(files) == 0:
        sys.exit("No Files found")
    for f in files:
        if f.endswith(".npy"):
            data = np.load(f,allow_pickle=True)
    return data 


if __name__ == "__main__":
    
    data = loadData()