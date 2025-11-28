from config import baseline_model_config,best_grid_searched_coefficient
import math

def get_scaled_coefficients(phi=0):
    """
        args: 
            phi: scaling factor for the efficient models varient
        returns:
            new_alpha,new_beta,new_gemma # model scaling coefficients
    """
    return pow(best_grid_searched_coefficient.alpha,phi), pow(best_grid_searched_coefficient.beta,phi), pow(best_grid_searched_coefficient.gemma,phi)



def get_varient_config(varient_type:str):
    valid_varient_types=['efficient_b0','efficient_b1','efficient_b2','efficient_b3','efficient_b4','efficient_b5','efficient_b6','efficient_b7']

    if varient_type not in valid_varient_types:
        raise ValueError(f"type: {varient_type} is not valid, choose from {valid_varient_types}")
    
    if varient_type==valid_varient_types[0]:
        scaled_coeff=get_scaled_coefficients(phi=0)
    elif varient_type==valid_varient_types[1]:
        scaled_coeff=get_scaled_coefficients(phi=1)
    elif varient_type==valid_varient_types[2]:
        scaled_coeff=get_scaled_coefficients(phi=2)
    elif varient_type==valid_varient_types[3]:
        scaled_coeff=get_scaled_coefficients(phi=3)
    elif varient_type==valid_varient_types[4]:
        scaled_coeff=get_scaled_coefficients(phi=4)
    elif varient_type==valid_varient_types[5]:
        scaled_ceoff=get_scaled_coefficients(phi=5)
    elif varient_type==valid_varient_types[6]:
        scaled_coeff=get_scaled_coefficients(phi=6)
    else:
        scaled_coeff=get_scaled_coefficients(phi=7)
    

    resolution_config=[math.ceil(v2[0]*scaled_coeff[2]) for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='r'] #resolution=gemma
    channel_config=[v2 for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='c'] # width==beta
    depth_config=[v2 for k,v in baseline_model_config.items() for k2,v2 in v.items() if k2=='l'] # depth =alpha

    print(resolution_config)


get_varient_config('efficient_b1')