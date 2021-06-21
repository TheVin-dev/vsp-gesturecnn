import numpy as np 
import os 
import numpy as np 
import sys
import cv2 
import time 
import torch 

# class Landmarks(torch.utils.data.Dataset):
#     def __init__(self, list_IDs, labels,data):
#         self.labels = labels
#         self.list_IDs = list_IDs   
#         self.data = data 
#     def __getitem__(self, index):
#         """
#         Generates one sample of data
#         """
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         # X = torch.load('data/' + ID + '.pt')
#         X = self.data[ID]
#         y = self.labels[ID]

#         return X, y


def loadAllData(labels,N=100):

    cwd = os.getcwd() 
    parentDir = os.path.dirname(cwd)
    dataDir = os.path.join(parentDir,"data")
    data = None # numpy array
    # and  "full" in f
    files = [[f for f in os.listdir(dataDir) if label in f and  f.endswith('.npy')] for label in labels]
    d = dict(zip(labels,files))
    print(d)

    if not any(files):
        sys.exit("No Files found")
    
    for label,listf in d.items():
        for f in listf:

            n = f.split("-",3)[2].split(".")[0]
            print(f"found {n} samples in {f}")

            if data is None:
                tmpdata = np.load(os.path.join(dataDir,f),allow_pickle=True)
                length = tmpdata.shape[0]
                if N < length: 
                    data = tmpdata[:N]
                else:
                    data = tmpdata
            else:
                d = np.load(os.path.join(dataDir,f),allow_pickle=True)
                length = data.shape[0]
                if N < length:
                    np.append(data,d[:N])
                    return data
                if data.shape[0] < N:
                    np.append(data,d)
    

    return data 

def CheckImages(data):

    """
    Helper function to look at data 
    """
   
    i=1
    while i < rawdata.shape[0]:
        
        data= rawdata[i][:]

       
        label, img = data
        print(f"showing {i} image")
        cv2.imshow('Loaded image',img)

        p =  cv2.waitKey(500)
        while p != 32:
            p =  cv2.waitKey(500)

            time.sleep(0.1)

        i+=1

    return None

def processData(data):
    #data = [[label,inputs],[label,inputs]]
        
    return data 
def loadSingleFile(name):


    cwd = os.path.dirname(__file__)
    parentDir =  os.path.dirname(cwd)
    dataDir = os.path.join(parentDir,"data")

    data = None # numpy array

    data = np.load(os.path.join(dataDir,name),allow_pickle=True)

    

    return data 

if __name__ == "__main__":
    
    data = loadSingleFile('test-100-1.npy')
   
    print(data[2])
    inputs = torch.utils.data.DataLoader(data,shuffle=True,batch_size=50)
    print(inputs)
    i1, l1 = next(iter(inputs))
    print(i1.shape)