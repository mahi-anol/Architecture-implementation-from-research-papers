from collections import namedtuple

ResNetConfig=namedtuple('ResNetConfig',['block','n_blocks','channels'])


resnet18_config=ResNetConfig(block='BasicBlock',
                            n_blocks=[2,2,2,2],
                            channels=[64,128,256,512],
                            ) 

resnet34_config=ResNetConfig(block='BasicBlock',
                            n_blocks=[3,4,6,3],
                            channels=[64,128,256,512],
                            )

resnet50_config=ResNetConfig(block='BottleNeck',
                            n_blocks=[3,4,6,3],
                            channels=[64,128,256,512],
                            )

resnet101_config=ResNetConfig(block='BottleNeck',
                            n_blocks=[3,4,23,3],
                            channels=[64,128,256,512],
                            )

resnet152_config=ResNetConfig(block='BottleNeck',
                            n_blocks=[3,8,36,3],
                            channels=[64,128,256,512],
                            ) 