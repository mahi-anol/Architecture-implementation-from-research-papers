import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
import torch
from config import baseline_model_config,kernel_configs
from utils import model_helpers,model_configs
from typing import OrderedDict


input_resolution,output_channel_configs,stage_repeats,stride_configs=model_configs.get_varient_configs()


class MBConvBlock(nn.Module):
    def __init__(self,expansion_ratio=1,re_ratio=0.25,in_channels=1,out_channels=1,input_image_size=None,stride=None,kernel_size=None):
        super().__init__()
        self.expansion_ratio=expansion_ratio
        self.block_stride=stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        # Expansion Layer
        if expansion_ratio!=1:
            self.expansion_layer=model_helpers.same_padded_conv2D(image_size=input_image_size)(in_channels=self.in_channels
                                                                                                ,out_channels=self.in_channels*expansion_ratio
                                                                                                ,kernel_size=1
                                                                                                ,stride=1 # for expansion stride is supposed to be 1.
                                                                                                ,dilation=1
                                                                                                ,groups=1
                                                                                                ,bias=False
                                                                                                ) ## expansion layer.  112x122x32-> 112x112*(32*expans
            self.bn0=nn.BatchNorm2d(num_features=in_channels*expansion_ratio)

        # Depth-wise Layer
        self.depth_wise_layer=model_helpers.same_padded_conv2D(image_size=input_image_size)(in_channels=self.in_channels*expansion_ratio
                                                                                            ,out_channels=self.in_channels*expansion_ratio
                                                                                            ,kernel_size=kernel_size
                                                                                            ,stride=self.block_stride
                                                                                            ,dilation=(1,1)
                                                                                            ,groups=in_channels*expansion_ratio
                                                                                            ,bias=False
                                                                                            )# 112,112,(32*expansion Ratio) -> 112,112,(32* expansion Ratio)
        self.bn1=nn.BatchNorm2d(num_features=expansion_ratio*in_channels)

        output_image_size=model_helpers.get_output_image_size(input_image_size=input_image_size,stride=self.block_stride)
        # Sqeeze and Excitation Layer (attention)
        reduced_channels=max(1,(expansion_ratio*in_channels)*re_ratio)
        self.reduce=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=(expansion_ratio*self.in_channels),out_channels=reduced_channels,kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=True)
        self.expand=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=reduced_channels,out_channels=(expansion_ratio*self.in_channels),kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=True) # (1,1)

        # pointwise Conv
        self.point_conv=model_helpers.same_padded_conv2D(image_size=output_image_size)(in_channels=expansion_ratio*self.in_channels,out_channels=self.out_channels,kernel_size=(1,1),stride=(1,1),dilation=(1,1),groups=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_features=out_channels)

    # Forward method.
    def forward(self,inputs,drop_connect_rate=None):

        x=inputs
        # Expansion Layer
        if self.expansion_ratio>1:
            x=self.expansion_layer(x)
            x=self.bn0(x)
            x=F.silu(x,inplace=False)

        # DepthWise Layer
        x=self.depth_wise_layer(x)  #halfs the spatial resolution
        x=self.bn1(x)
        x=F.silu(x,inplace=False)

        ### Sqeeze and Excitation layer
        before_pool=x
        x=F.adaptive_avg_pool2d(x,(1,1))
        x=self.reduce(x)
        x=F.silu(x,inplace=False)
        x=self.expand(x)
        x=F.sigmoid(x,inplace=False)*before_pool #(b,c,1,1)*b,c,(h,W)  = b,c,h,w

        # projection/pointwise conv
        x=self.point_conv(x)
        x=self.bn2(x)
        
        # skip connection
        if self.in_channels==self.out_channels and self.block_stride==1:
            if drop_connect_rate:
                x=model_helpers.drop_connect(inputs=x,p=drop_connect_rate,training=self.training)
            inputs+=x

        return inputs
    

class stem_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv3x3=model_helpers.same_padded_conv2D(image_size=input_resolution)(in_channels=3
                                                                                    ,out_channels=output_channel_configs[0]
                                                                                    ,kernel_size=kernel_configs[0]
                                                                                    ,stride=stride_configs[0]
                                                                                    ,dilation=(1,1)
                                                                                    ,groups=1
                                                                                    ,bias=False
                                                                                    )
        self.bn=nn.BatchNorm2d(num_features=output_channel_configs[0])


    def forward(self,x):
        x=self.conv3x3(x)
        x=self.bn(x)
        x=F.silu(x,inplace=False)
        return x  


class final_bootleneck_layer(nn.Module):
    def __init__(self,image_size,in_channels,out_channels,kernel_size,stride,final_classes=10):
        self.conv1x1=model_helpers.same_padded_conv2D(image_size=image_size)(in_channels=in_channels
                                                                            ,out_channels=out_channels
                                                                            ,kernel_size=kernel_size
                                                                            ,stride=stride
                                                                            ,dilation=1
                                                                            ,groups=1
                                                                            ,bias=False
                                                                            )
        self.bn=nn.BatchNorm2d(num_features=out_channels)
        
        self.pooling=nn.AdaptiveAvgPool1d(1)

        self.flattend=nn.Linear(in_channels=out_channels,out_channels=final_classes,bias=True)

    def forward(self,x):
        x=self.conv1x1(x)
        x=self.bn(x)
        x=F.silu(x,inplace=False)
        x=self.pooling(x)
        x=torch.flatten(x,1)
        x=self.flattend(x)
        return x

class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem_layer=stem_layer()
        output_image_shape= model_helpers.get_output_image_size(input_resolution,stride_configs[0])

        
        self.mb_conv_layers=[] # will contain MBconv layers of stage 2-8

    
        # stage 2-8 repeat configs
        repeat_configs=stage_repeats[1:-1]  # First and last block/layer same regard less of scaling.

        stage_wise_blocks=dict()

        for stage,repeat in enumerate(repeat_configs,1):
            for i in range(repeat):
                self.mb_conv_layers.append (MBConvBlock(expansion_ratio= 1 if stage==1 else 6
                                                        ,re_ratio=0.25
                                                        ,in_channels=output_channel_configs[stage-1]
                                                        ,out_channels=output_channel_configs[stage]
                                                        ,input_image_size=output_image_shape
                                                        ,stride=stride_configs[stage]
                                                        ,kernel_size=kernel_configs[stage]
                                                        )
                                            )
            output_image_shape=model_helpers.get_output_image_size(output_image_shape,stride_configs[stage])


        self.final_BottleNeck=final_bootleneck_layer(image_size=output_image_shape
                                                     ,in_channels=output_channel_configs[-2]
                                                     ,out_channels=output_channel_configs[-1]
                                                     ,kernel_size=1
                                                     ,stride=stride_configs[-1]
                                                     ,final_classes=10
                                                    )        

    def forward(self,inputs):

        # Stem layer Forward pass
        x=self.stem_layer(inputs)

        # Mbconv Layers Forward Pass
        for layer in self.mb_conv_layers:
            x=layer(x,0.3)

        # Final Layer forward pass
        x=self.final_BottleNeck(x)
        return x
    

if __name__=="__main__":
    model=EfficientNet()
    print(model)
    
    inputs=torch.rand((1,3,224,224))
    print(model(inputs))
