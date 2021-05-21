# This module contains all the models we defined 
import torch
from torch import optim
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# This model performs each digit classification with 2 different CNNs   
class Base_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layers that handle digit classification with first CNN
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1_1 = nn.Linear(256, 200)
        self.fc2_1 = nn.Linear(200, 10)
        
        # Layers that handle digit classification with second CNN
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1_2 = nn.Linear(256, 200)
        self.fc2_2 = nn.Linear(200, 10)
        
        # Layers that handle comparisson 
        self.fc3 = nn.Linear(20, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 2)
        
    def cnn1(self, x):
        x = F.relu(F.max_pool2d(self.conv1_1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_1(x), kernel_size=2))
        x = F.relu(self.fc1_1(x.view(-1, 256)))
        x = self.fc2_1(x)
        return x
    
    def cnn2(self, x):
        x = F.relu(F.max_pool2d(self.conv1_2(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_2(x), kernel_size=2))
        x = F.relu(self.fc1_2(x.view(-1, 256)))
        x = self.fc2_2(x)
        return x
    
    def mlp(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        s = x.shape
        input_1 = x[:,0,:,:].reshape([s[0],1,s[2],s[3]])
        input_2 = x[:,1,:,:].reshape([s[0],1,s[2],s[3]])
        
        output_1 = self.cnn1(input_1)
        output_2 = self.cnn2(input_2)
        
        concatenated = torch.cat((output_1, output_2), 1)
        
        comparison = self.mlp(concatenated)
        return comparison   
    
# Classification Model using wheight sharing 
  
class Weight_Sharing_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layers that handle digit classification 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # Layers that handle comparisson 
        self.fc3 = nn.Linear(20, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 2)
        
    def cnn(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def mlp(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        s = x.shape
        input_1 = x[:,0,:,:].reshape([s[0],1,s[2],s[3]])
        input_2 = x[:,1,:,:].reshape([s[0],1,s[2],s[3]])
        
        output_1 = self.cnn(input_1)
        output_2 = self.cnn(input_2)
        
        concatenated = torch.cat((output_1, output_2), 1)
        
        comparison = self.mlp(concatenated)
        return comparison

# Classification Model using wheight sharing and an auxiliary loss 

class Auxiliary_Loss_Weight_Sharing_Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layers that handle digit classification 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # Layers that handle comparisson 
        self.fc3 = nn.Linear(20, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 2)
        
    def cnn(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
    
    def mlp(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        s = x.shape
        
        input_1 = x[:,0,:,:].reshape([s[0],1,s[2],s[3]])
        input_2 = x[:,1,:,:].reshape([s[0],1,s[2],s[3]])
        
        output_1 = self.cnn(input_1)
        output_2 = self.cnn(input_2)
        
        concatenated = torch.cat((output_1, output_2), 1)
        
        comparison = self.mlp(concatenated)
        return output_1, output_2, comparison  
    
# Same auxiliary loss but adding some Dropout Layers during Digit Classification

class Auxiliary_Loss_Net_Dropout(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layers that handle digit classification 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.dropout_1 = nn.Dropout(p=0.1)
        self.dropout_2 = nn.Dropout(p=0.1)
        self.dropout_3 = nn.Dropout(p=0.1)
        # Layers that handle comparisson 
        self.fc3 = nn.Linear(20, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 2)
        
    def cnn(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = self.dropout_1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.dropout_2(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout_3(x)
        x = self.fc2(x)
        return x
    
    def mlp(self, x):
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def forward(self, x):
        s = x.shape
        
        input_1 = x[:,0,:,:].reshape([s[0],1,s[2],s[3]])
        input_2 = x[:,1,:,:].reshape([s[0],1,s[2],s[3]])
        
        output_1 = self.cnn(input_1)
        output_2 = self.cnn(input_2)
        
        concatenated = torch.cat((output_1, output_2), 1)
        
        comparison = self.mlp(concatenated)
        return output_1, output_2, comparison  
    