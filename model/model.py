import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch # 추가

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

  
class AlexNet(nn.Module):
  def __init__(self):
      super(AlexNet, self).__init__()
      # Image input_size=(3, 227, 227)
      self.layers = nn.Sequential(
          # input_size=(96, 55, 55)
          nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4, padding=0), 
          nn.ReLU(), 
          # input_size=(96, 27, 27)
          nn.MaxPool2d(kernel_size=3, stride=2),
          # input_size=(256, 27, 27)
          nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),
          nn.ReLU(),
          # input_size=(256, 13, 13)
          nn.MaxPool2d(kernel_size=3, stride=2), 
          # input_size=(384, 13, 13)
          nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
          nn.ReLU(),
          # input_size=(384, 13, 13)
          nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
          nn.ReLU(),
          # input_size=(256, 13, 13)
          nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
          nn.ReLU(),
          # input_size=(256, 6, 6)
          nn.MaxPool2d(kernel_size=3, stride=2), 
      )
      self.classifier = nn.Sequential(
          nn.Dropout(p=0.5),
          nn.Linear(in_features=256*6*6, out_features=4096),
          nn.ReLU(),
          nn.Dropout(p=0.5),
          nn.Linear(in_features=4096, out_features=4096),
          nn.ReLU(),
          nn.Linear(in_features=4096, out_features=1000),
      )
  
  def forward(self, x):
      x = self.layers(x)
      x = x.view(-1, 256*6*6)
      x = self.classifier(x)
      return x

