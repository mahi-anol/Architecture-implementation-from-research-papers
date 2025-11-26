import torch.nn as nn
from dataclasses import dataclass

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=(2,2),padding=(1,1)) # (224,224,3) -> (112,112,32)
        self.batchNorm1=nn.BatchNorm2d(num_features=32)
        
class MBConvBlock(nn.Module):
    def __init__(self,expansion_ratio=1):
        super().__init__()
        self.layers=[]
        
        # Expansion Layer
        if expansion_ratio!=1:
           self.expansion_layer=nn.Conv2d(in_channels=32,out_channels=expansion_ratio*32,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False) ## expansion layer. 
           self.layers.append(self.expansion_layer)
        
        # Depth-wise Layer
        self.depth_wise_layer=nn.Conv2d(in_channels=expansion_ratio*32,out_channels=(expansion_ratio*32)//2,groups=expansion_ratio*32,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False) # 112,112,32 -> 112,112,16
        self.layers.append(self.depth_wise_layer)

        # Sqeeze and Excitation Layer
        



