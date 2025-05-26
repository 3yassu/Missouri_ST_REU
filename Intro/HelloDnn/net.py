#net.py
import torch.nn as nn

class Net(nn.Module): #Neural Network Class
    def __init__(self): #
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)#Takes the image, has a filter (of 3x3 pixels) starts from the top left and
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x): #
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  #Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
