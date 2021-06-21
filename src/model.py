import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader,TensorDataset
import torch 
import torch.optim as optim
import loadData 
import numpy as np 

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 
        self.fc1 = nn.Linear(4*(42+11),120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self,x):

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return  torch.sigmoid(self.fc3(x))


if __name__== "__main__":
    data = loadData.loadSingleFile("test-100-1.npy")
    net =Net()

    labels = [] 
    listinputs=[] 
    for idx, data in enumerate(data):
        label, inputs = data
        if len(inputs) != 212:
            continue
        labels.append(label)
        listinputs.append(inputs)
    
    #for data in listinputs:
    #     print(type(data),len(data),"\n",data,"\n")
    #     input("") 

    trainloader = torch.utils.data.DataLoader(traindata, shuffle=True, batch_size=50)
    print(dir(trainloader))
    for idx, (label,data) in enumerate(trainloader):
        print(idx,label,data)
