import torch
import torch.nn as nn
from utils import align_outputs

class UNET(nn.Module):
    def __init__(self,in_channel=1,out_Channels=2):
        super(UNET,self).__init__()
        channel_config=[in_channel,64,128,256,512,1024]
        
        # len =6 and we need 4 channel expansion transformation throughout the encoder. so loop needs 4 iter.
        self.encoder_layers=nn.ModuleList()
        for i in range(len(channel_config)-2):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channel_config[i],out_channels=channel_config[i+1],kernel_size=(3,3)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=channel_config[i+1],out_channels=channel_config[i+1],kernel_size=(3,3)),
                    nn.ReLU(),
                ),
            )
            self.encoder_layers.append(
                    nn.MaxPool2d(kernel_size=(2,2))
            )

        self.bottle_neck_layer=nn.Sequential(
            nn.Conv2d(in_channels=channel_config[-2],out_channels=channel_config[-1],kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_config[-1],out_channels=channel_config[-1],kernel_size=(3,3)),
            nn.ReLU(),
        )

        self.decoder_layers=nn.ModuleList()

        for i in range(len(channel_config)-1,1,-1):
            self.decoder_layers.append(
                nn.ConvTranspose2d(in_channels=channel_config[i],out_channels=channel_config[i-1],kernel_size=(2,2),stride=(2,2)),
            )
            self.decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=channel_config[i],out_channels=channel_config[i-1],kernel_size=(3,3)),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=channel_config[i-1],out_channels=channel_config[i-1],kernel_size=(3,3)),
                    nn.ReLU(),
                )
            )
        self.output_layer=nn.Conv2d(in_channels=channel_config[1],out_channels=out_Channels,kernel_size=(1,1))

    
    def forward(self,inp):
        x=inp
        encoder_layerwise_output=[]
        for layer in self.encoder_layers:
            # pass through the layer, whether its maxpool or sequential.
            x=layer(x)
            #only store sequential layers output.
            if isinstance(layer,nn.Sequential):
                encoder_layerwise_output.append(x)
        # print("Encoder output: ",x.shape)
        x=self.bottle_neck_layer(x)

        # print("BottleNeck Output: ",x.shape)
        len_encoder_stage=len(encoder_layerwise_output)
        # print(encoder_layerwise_output[-1].shape)
        idx_enc=1
        for layer in self.decoder_layers:
            if isinstance(layer,nn.ConvTranspose2d):
                x=layer(x)
            elif isinstance(layer,nn.Sequential):
                encoder_out=encoder_layerwise_output[len_encoder_stage-idx_enc]
                # print(encoder_out.shape)
                # print(x.shape)
                combined_out=align_outputs(encoder_out,x)
                # print(combined_out.shape)
                x=layer(combined_out)
                # print("Decoder stage: ",x.shape)
                idx_enc+=1

        x=self.output_layer(x)
        return x
    
if __name__=="__main__":
    test_input=torch.randn((4,1,572,572))

    model=UNET(1,3)

    print(model)
    output=model(test_input)
    print(output.shape)
