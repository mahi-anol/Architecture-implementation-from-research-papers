import torch.nn as nn
from torchinfo import summary

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1,downsample=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(num_features=out_channels)
        self.relu=nn.ReLU(inplace=False)
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=stride,bias=False)
            bn=nn.BatchNorm2d(num_features=out_channels)
            downsample=nn.Sequential(conv,bn)
        else:
            downsample=None
        
        self.downsample=downsample

    def forward(self,x):
        i=x

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        if self.downsample is not None:
            i=self.downsample(i)
        x+=i
        return x

class BottleNeck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1,downsample=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(num_features=out_channels)
        self.relu=nn.ReLU(inplace=False)

        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_features=out_channels)
        
        self.conv3=nn.Conv2d(in_channels=out_channels,out_channels=self.expansion*out_channels,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(num_features=self.expansion*out_channels)
        if downsample:
            conv=nn.Conv2d(in_channels=in_channels,out_channels=self.expansion*out_channels,kernel_size=1,stride=stride,bias=False)
            bn=nn.BatchNorm2d(num_features=self.expansion*out_channels)
            downsample=nn.Sequential(conv,bn)
        else:
            downsample=None

        self.downsample=downsample



    def forward(self,x):
        i=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)

        x=self.conv3(x)
        x=self.bn3(x)
        if self.downsample is not None:
            i=self.downsample(i)
        x+=i
        x=self.relu(x)
        return x



class ResNet(nn.Module):
    def __init__(self,config,outdim):
        super().__init__()
        block,n_blocks,channels=config
        block=BasicBlock if block=='BasicBlock' else BottleNeck

        self.in_channels=channels[0]
        assert len(n_blocks) == len(channels)==4

        ## stem layer
        self.conv1=nn.Conv2d(in_channels=3,out_channels=self.in_channels,kernel_size=(7,7),stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.relu=nn.ReLU(inplace=False)
        self.maxpool2d=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        ### first block
        self.layer1=self.get_resnet_layer(block,n_blocks[0],channels[0])
        ### second block
        self.layer2=self.get_resnet_layer(block,n_blocks[1],channels[1])
        ### third block
        self.layer3=self.get_resnet_layer(block,n_blocks[2],channels[2])
        ### 4th layer
        self.layer4=self.get_resnet_layer(block,n_blocks[3],channels[3])

        ### avgpool
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(self.in_channels,outdim)


    def get_resnet_layer(self,block,n_blocks,channels,stride=1):
        layers=[]
        if self.in_channels != block.expansion*channels:
            downsample=True
        else:
            downsample=False

        layers.append(block(self.in_channels,channels,stride,downsample))
        for i in range(1,n_blocks):
            layers.append(block(block.expansion*channels,channels))
            self.in_channels=block.expansion*channels
        
        return nn.Sequential(*layers)


    def forward(self,x):
        # Stem layer forward pass
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool2d(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        h=x.view(x.shape[0],-1) # Flatten
        x=self.fc(h)
        return x
    

if __name__=="__main__":
    from config import resnet50_config,resnet152_config
    import torch


    model=ResNet(config=resnet50_config,outdim=10)
    inputs=torch.rand((2,3,224,224))
    print('Test input shape: ',inputs.shape)
    output=model(inputs)
    print('Model output shape: ',output.shape)

    ### show compelte model insights
    summary(model,input_size=(1,3,224,224))