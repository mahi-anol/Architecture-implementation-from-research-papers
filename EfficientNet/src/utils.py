from config import baseline_model_config,best_grid_searched_coefficient,tpu_friendly_efficient_resolutions
import math
from functools import partial
from torch import nn

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
    def get_output_image_size(input_image_size,stride):
        # h and w for input image
        h,w=(input_image_size,input_image_size) if isinstance(input_image_size,int) else input_image_size
        stride=stride if isinstance(stride,int) else stride[0]
        image_height=int(math.ceil(image_height/stride))
        image_width=int(math.ceil(image_width/stride))
        return image_height,image_width
    
    @staticmethod
    def same_padded_conv_2d(image_size=None):
        return model_helpers.dynamically_same_padded_conv2D if image_size is None else partial(model_helpers.statically_same_padded_conv2D,image_size)
    
    @staticmethod
    def dynamically_same_padded_conv2D(nn.conv2d):
        pass



    # @staticmethod
    # def statically_same_padded_conv2D():
    #     pass


if __name__=="__main__":
    configs=model_configs()
    configs.get_varient_configs('efficient_b4')