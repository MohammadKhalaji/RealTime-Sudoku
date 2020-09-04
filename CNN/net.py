import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module): 
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=12*12*64, out_features=128)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=9)
        
    def forward(self, x): 
        t = self.conv1(x)
        t = F.relu(t)
        
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = self.dropout1(t)
        
        
        t = t.view(t.shape[0], -1)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.dropout2(t)
        t = self.fc2(t)                                                                                                                                              
        
        return F.log_softmax(t, dim=1)