import collections

### This config is mentioned in the paper and was achieved through NAS.

baseline_model_config={
    'stage1': {
        'r':(224,224), # Resolution (Hi,wi)
        'c':32, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage2': {
        'r':(112,112), # Resolution (Hi,wi)
        'c':16, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage3': {
        'r':(112,112), # Resolution (Hi,wi)
        'c':24, # output channel (ci)
        'l':2, #(li) number of repeat for a specific operator.
    },
    'stage4': {
        'r':(56,56), # Resolution (Hi,wi)
        'c':40, # output channel (ci)
        'l':2, #(li) number of repeat for a specific operator.
    },
    'stage5': {
        'r':(28,28), # Resolution (Hi,wi)
        'c':80, # output channel (ci)
        'l':3, #(li) number of repeat for a specific operator.
    },
    'stage6': {
        'r':(14,14), # Resolution (Hi,wi)
        'c':112, # output channel (ci)
        'l':3, #(li) number of repeat for a specific operator.
    },
    'stage7': {
        'r':(14,14), # Resolution (Hi,wi)
        'c':192, # output channel (ci)
        'l':4, #(li) number of repeat for a specific operator.
    },
    'stage8': {
        'r':(7,7), # Resolution (Hi,wi)
        'c':320, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
    'stage9': {
        'r':(7,7), # Resolution (Hi,wi)
        'c':1280, # output channel (ci)
        'l':1, #(li) number of repeat for a specific operator.
    },
}

best_grid_searched_coefficient=collections.namedtuple(typename="Best_Coefficients",field_names=['alpha','beta','gemma'])(alpha=1.2,beta=1.1,gemma=1.15) # Directly Taken from the paper.


