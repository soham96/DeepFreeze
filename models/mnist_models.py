import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1=nn.Linear(784, 128)
        self.fc2=nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
        self.name = 'Base.pt'
        self.conv1 = nn.Conv2d(1, 10, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, padding=2)
        self.fc1 = nn.Linear(20*28*28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x=torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class BaseWide(nn.Module):
    def __init__(self):
        super(BaseWide, self).__init__()
        self.name = 'BaseWide.pt'
        self.conv1 = nn.Conv2d(1, 20, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(20, 40, 5, 1, padding=2)
        self.fc1 = nn.Linear(40*28*28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x=torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class BaseDropout(nn.Module):
    def __init__(self):
        super(BaseDropout, self).__init__()
        self.name = 'BaseDropout.pt'
        self.conv1 = nn.Conv2d(1, 10, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(10, 20, 5, 1, padding=2)
        self.drop_1=nn.Dropout(0.5)
        self.drop_2=nn.Dropout(0.5)
        self.fc1 = nn.Linear(20*28*28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.drop_1(x))
        x=torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x=F.relu(self.drop_2(x))
        x = self.fc2(x)

        return x

class BasePReLU(nn.Module):
    def __init__(self):
        super(BasePReLU, self).__init__()
        self.name = 'BasePReLU.pt'
        self.conv1 = nn.Conv2d(1, 10, 5, 1, padding=2)
        self.prelu1=nn.PReLU()

        self.conv2 = nn.Conv2d(10, 20, 5, 1, padding=2)
        self.prelu2=nn.PReLU()

        self.fc1 = nn.Linear(20*28*28, 50)
        self.prelu3=nn.PReLU()

        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x=torch.flatten(x, 1)
        x = self.prelu3(self.fc1(x))
        x = self.fc2(x)

        return x

class BaseDBNorm(nn.Module):
    def __init__(self):
        super(BaseDBNorm, self).__init__()
        self.name = 'BaseDBNorm.pt'
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 10, 5, 1)
        self.batchnormconv1 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 20, 5, 1)
        self.conv4 = nn.Conv2d(20, 20, 5, 1)
        self.batchnormconv2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(20*12*12, 50)
        self.batchnormdense1= nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #if not self.training:
        #    import ipdb; ipdb.set_trace()
        x = F.relu(self.conv1(x))
        x = F.relu(self.batchnormconv1(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.batchnormconv2(self.conv4(x)))
        x=torch.flatten(x, 1)
        x = F.relu(self.batchnormdense1(self.fc1(x)))
        #np.save('layer_input.npy', x.cpu().detach().numpy())
        x = self.fc2(x)

        return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.name = 'LeNet5.pt'
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, padding=2)
        self.conv4 = nn.Conv2d(120, 120, 5, 1, padding=1)
        self.fc0 = nn.Linear(120, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class LeNet5DBNorm(nn.Module):
    def __init__(self):
        super(LeNet5DBNorm, self).__init__()
        self.name = 'LeNet5Dropout.pt'
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.batchnormconv1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.batchnormconv2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        self.batchnormconv3 = nn.BatchNorm2d(120)
        self.drop_1=nn.Dropout(0.5)
        self.fc1 = nn.Linear(120*16*16, 84)
        self.drop_2=nn.Dropout(0.5)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        #if not self.training:
        #    import ipdb; ipdb.set_trace()
        x = F.max_pool2d(F.relu(self.batchnormconv1(self.conv1(x))))
        x = F.max_pool2d(F.relu(self.batchnormconv2(self.conv2(x))))
        x = F.relu(self.batchnormconv3(self.conv3(x)))
        x = F.relu(self.drop_1(x))
        x=torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.drop_2(x))
        #np.save('layer_input.npy', x.cpu().detach().numpy())
        x = self.fc2(x)

        return x

class LeNet5Dropout(nn.Module):
    def __init__(self):
        super(LeNet5Dropout, self).__init__()
        self.name = 'LeNet5Dropout.pt'
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, padding=2)
        self.conv3 = nn.Conv2d(16, 120, 5, 1, padding=2)
        self.conv4 = nn.Conv2d(120, 120, 5, 1, padding=1)
        self.fc0= nn.Linear(120, 120)
        self.drop_1=nn.Dropout(0.5)
        self.fc1 = nn.Linear(120, 84)
        self.drop_2=nn.Dropout(0.5)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x= F.max_pool2d(F.relu(self.drop_1(x)), 2)
        x = F.relu(self.conv4(x))
        x=torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.drop_2(x))
        x = self.fc2(x)

        return x
