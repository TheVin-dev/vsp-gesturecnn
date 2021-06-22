import torch.nn as nn 
import torch.nn.functional as F 
import torch 
import torch.optim as optim
import numpy as np 
import data_utils   
from data_utils import MyDataLoader
import os 
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() 
        self.fc1 = nn.Linear(4*(42+11),120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,3)
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.to(device)
    
    def forward(self,x):

        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return  torch.sigmoid(self.fc3(x))


def train(trainloader,model,epoch=5):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    epochs = epoch
    steps = 0
    running_loss = 0
    print_every = 50
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
          
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) #criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in trainloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(trainloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(trainloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(trainloader):.3f}")
                running_loss = 0
            model.train()
    return model


if __name__== "__main__":
    data = MyDataLoader("full-600.npy")
    trainloader = torch.utils.data.DataLoader(data,shuffle=True,batch_size=10)
    
    model = train(trainloader,Net(),4)
    torch.save(model,os.path.join(data_utils.modelDir,"model.pth"))