import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
import torch
class MBConvBlock(nn.Module):
    def __init__(self,expansion_ratio=4,reduce_ratio=4):
        super().__init__()
        self.expansion_ratio=expansion_ratio
        
        # Expansion Layer
        if expansion_ratio!=1:
            self.expansion_layer=nn.Conv2d(in_channels=32,out_channels=expansion_ratio*32,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False) ## expansion layer.  112x122x32-> 112x112*(32*expans
            self.bn0=nn.BatchNorm2d(num_features=expansion_ratio*32)

        # Depth-wise Layer
        self.depth_wise_layer=nn.Conv2d(in_channels=expansion_ratio*32,out_channels=(expansion_ratio*32),groups=expansion_ratio*32,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False) # 112,112,(32*expansion Ratio) -> 112,112,(32* expansion Ratio)
        self.bn1=nn.BatchNorm2d(num_features=expansion_ratio*32)

        # Sqeeze and Excitation Layer (attention)
        self.global_avg_pool=nn.AdaptiveAvgPool2d(output_size=(1,1)) # (1,1,channel=32*expansion)
        self.reduce=nn.Conv2d(in_channels=(expansion_ratio*32),out_channels=expansion_ratio*32//reduce_ratio,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)
        self.expand=nn.Conv2d(in_channels=(expansion_ratio*32)//reduce_ratio,out_channels=(expansion_ratio*32),kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True) # (1,1)

        # pointwise Conv
        self.point_conv=nn.Conv2d(in_channels=expansion_ratio*32,out_channels=expansion_ratio*32//2,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=False)
        self.bn2=nn.BatchNorm2d(num_features=expansion_ratio*32//2)

    # Forward method.
    def forward(self,inputs):
        x=inputs

        # Expansion Layer
        if self.expansion_ratio>1:
            x=self.expansion_layer(x)
            x=self.bn0(x)
            x=F.silu(x,inplace=False)

        # DepthWise Layer
        x=self.depth_wise_layer(x)
        x=self.bn1(x)
        x=F.silu(x,inplace=False)

        ### Sqeeze and Excitation layer
        before_pool=x
        x=self.global_avg_pool(x)
        x=self.reduce(x)
        x=F.silu(x,inplace=False)
        x=self.expand(x)
        x=F.sigmoid(x,inplace=False)*before_pool #(b,c,1,1)*b,c,(h,W)  = b,c,h,w

        # projection/pointwise conv
        x=self.point_conv(x)
        x=self.bn2(x)
        
        # skip connection
        if x.shape==inputs.shape:
            x+=inputs

        return x








class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Stem Layer
        self.conv3x3=nn.Conv2d(in_channels=3,out_channels=output_channel[0],kernel_size=(3,3),stride=(2,2),padding='same',bias=False) # (224,224,3) -> (112,112,32)
        self.batchNorm=nn.BatchNorm2d(num_features=output_channel[0])

    def forward(self,inputs):
        x=self.conv3x3(inputs)
        x=self.batchNorm(x)
        x=F.silu(x,inplace=False)
        return x
    

if __name__=="__main__":
    from utils import model_configs
    size,output_channel,repeat=model_configs().get_varient_configs('efficient_b0')
    print(size)
    print(output_channel)
    print(repeat)

    model=EfficientNet()
    
    inputs=torch.rand((1,3,224,224))
    print(model(inputs))
