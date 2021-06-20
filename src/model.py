import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    def forward(self,x):

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return  torch.sigmoid(self.fc3(x))
    
if __name__== "__main__":
    net = Net() 
