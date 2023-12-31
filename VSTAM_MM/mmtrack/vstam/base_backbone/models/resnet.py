import torch
import torch.nn as nn
from ..modules.identify import IdentifyModel

class Resnet50Custom(nn.Module):
    def __init__(self, pretrained=True, **kwargs):
        super(Resnet50Custom, self).__init__(**kwargs)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', "resnet50", pretrained=pretrained)
        # change first conv layer in last layer
        self.resnet.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(2, 2), dilation=2, bias=False)
        # this shortcut added to ouput, so lets change it too
        self.resnet.layer4[0].downsample = nn.Conv2d(1024, 2048, kernel_size=1, bias=False)
        # change avg_pool and fc layer to id operation
        self.resnet.avgpool = IdentifyModel()
        self.resnet.fc = IdentifyModel()

        self.strides = 16
        self.feature_embedder_out = 2048

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x