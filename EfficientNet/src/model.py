import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3=nn.Conv2d(3,32,(3,3),stride=2,padding=1) # 224x224 -> 112x112



class MBConvBlock(nn.Module):
    