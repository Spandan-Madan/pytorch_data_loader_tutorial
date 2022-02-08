import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
    ) -> None:
        super(SimpleCNN, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,224,1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.AvgPool2d(4)),
            ('conv2', nn.Conv2d(224,32,1)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.AvgPool2d(16)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(288,num_classes)),
        ]))
    def forward(self, x):
        return self.model(x)
