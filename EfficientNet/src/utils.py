from config import baseline_model_config,best_grid_searched_coefficient,tpu_friendly_efficient_resolutions
import math
from functools import partial
from torch import nn
from torch.nn import functional as F

class model_configs:
    @staticmethod
    def fixing_width(beta,value,divisor=8):
        scaled=beta*value
        fixed_width=max(divisor,(scaled+divisor/2)//divisor * divisor)

        if fixed_width<0.9*scaled:
            fixed_width+=divisor
        return int(fixed_width)

    @staticmethod
    def fixing_depth(alpha,value):
        scaled=alpha*value
        fixed_depth=math.ceil(scaled)
        return int(fixed_depth)


    def get_varient_configs(self,varient_type:str):

        valid_varient_types=['efficient_b0','efficient_b1','efficient_b2','efficient_b3','efficient_b4','efficient_b5','efficient_b6','efficient_b7']

        if varient_type not in valid_varient_types:
            raise ValueError(f"type: {varient_type} is not valid, choose from {valid_varient_types}")

        config_index=valid_varient_types.index(varient_type)

        resolution=tpu_friendly_efficient_resolutions[config_index]
        alpha=best_grid_searched_coefficient.alpha[config_index]
        beta=best_grid_searched_coefficient.beta[config_index]

        channel_configs=[self.fixing_width(beta,v2) for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='c'] # width==beta
        depth_configs=[self.fixing_depth(alpha,v2) for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='l'] # depth =alpha
        
        if __name__=="__main__":
            print(resolution)
            print(channel_configs)
            print(depth_configs)

        return resolution,channel_configs,depth_configs



class model_helpers:
    @staticmethod
    def get_output_image_size(input_image_size:tuple|int,stride:tuple|int):
        # h and w for input image
        ih,iw=(input_image_size,input_image_size) if isinstance(input_image_size,int) else input_image_size
        sh,sw=stride if isinstance(stride,tuple) else stride,stride
        image_height=int(math.ceil(ih/sh))
        image_width=int(math.ceil(iw/sw))
        return image_height,image_width
    
    @staticmethod
    def same_padded_conv_2d(image_size=None):
        return model_helpers.dynamically_same_padded_conv2D if image_size is None else partial(model_helpers.statically_same_padded_conv2D,image_size)
    

    class dynamically_same_padded_conv2D(nn.Conv2d):
        def __init__(self,in_channels,out_channels,kernel_size,stride=1,dialation=1,groups=1,bias=True):
            super().__init__(in_channels=in_channels
                             ,out_channels=out_channels
                             ,kernel_size=kernel_size
                             ,stride=stride
                             ,dialation=dialation
                             ,groups=groups
                             ,bias=bias
                            )
        def forward(self, inputs):
            ih,iw=inputs.size()[-2:] # b,c,h,w -> h,w
            kh,kw=self.weight.size()[-2:] #filter,channel,h,w
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation[0]+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation[1]+1-iw,0)
            
            if pad_h>0 or pad_w>0:
                inputs=F.pad(inputs,[pad_w//2,pad_w-pad_w//2,pad_h//2,pad_h-pad_h//2])
            return F.conv2d(input=inputs
                            ,weight=self.weight
                            ,bias=self.bias
                            ,stride=self.stride
                            ,padding=self.padding
                            ,dilation=self.dilation
                            ,groups=self.groups
                            )
            


    class statically_same_padded_conv2D(nn.Conv2d):
        def __init__(self,in_channels,out_channels,kernel_size,stride=1,image_size=None,**kwargs):
            super().__init__(in_channels=in_channels
                             ,out_channels=out_channels
                             ,kernel_size=kernel_size
                             ,stride=stride
                             ,**kwargs
                            )

            assert image_size is not None

            ih,iw=image_size if len(image_size)>1 else (image_size,image_size) # b,c,h,w -> h,w
            kh,kw=self.weight.size()[-2:] #filter,channel,h,w
            sh,sw=self.stride
            oh,ow=model_helpers.get_output_image_size((ih,iw),(sh,sw))
            pad_h=max((oh-1)*sh+(kh-1)*self.dilation[0]+1-ih,0)
            pad_w=max((ow-1)*sw+(kw-1)*self.dilation[1]+1-iw,0)
            if  pad_h>0 or pad_w>0:
                self.static_padding=nn.ZeroPad2d((pad_w//2,pad_w-pad_w//2
                                                  ,pad_h//2,pad_h-pad_h//2))
            else:
                self.state_padding=nn.Identity()
        
        def forward(self,inputs):
            inputs=self.state_padding(inputs)
            inputs=F.conv2d(input=inputs
                            ,weight=self.weight
                            ,bias=self.bias
                            ,stride=self.stride
                            ,padding=self.padding
                            ,dilation=self.dilation
                            ,groups=self.groups
                            )
            
    @staticmethod
    def same_padded_maxpool2D(image_size=None):
        if image_size is None:
            return dynamically_same_padded_maxpool2D
        else:
            return statically_same_padded_maxpool2D
        
    
    class dynamically_same_padded_maxpool2D(nn.MaxPool2d):

        def __init__(self,kernel_size,stride,padding=0,dialation=1,return_indices=False,ceil_mode=False):
            super().__init__(kernel_size=kernel_size,stride=stride,padding=padding,dialation=dialation,return_indices=return_indices,ceil_mode=ceil_mode)


        def forward(self,inputs):
            ih,iw=inputs.size()[-2:]
            kh,kw=self.kernel_size
            sh,sw=self.stride
            oh,ow=
        


if __name__=="__main__":
    configs=model_configs()
    configs.get_varient_configs('efficient_b4') 