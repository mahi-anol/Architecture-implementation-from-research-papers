from logger import logging
import torch
from exception import MyException
import os,sys
import torch.nn as nn

def saving_model_with_state_and_logs(model,optimizer,results,file="model.pt"):
    try:
        os.makedirs('checkpoints',exist_ok=True)
        path=os.path.join('checkpoints',file)
        contents={
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'history':results,
        }
        torch.save(contents,path)
    except Exception as e:
        logging.error("There was an unexpect error during saving the model artifacts")
        raise MyException(e,sys)
    else:
        logging.info(f"Succesfully saved the model artifact at {path}")

# cropping
def align_outputs(encoder_output,prev_layer_output):

    #Negative_output
    _,_,enc_out_h,enc_out_w=encoder_output.shape
    _,_,prev_out_h,prev_out_w=prev_layer_output.shape
    
    # Height croping
    prev_out_center_h=prev_out_h//2
    enc_out_center_h=enc_out_h//2
    crop_h=enc_out_center_h-prev_out_center_h


    # width cropping
    prev_out_center_w=prev_out_w//2
    enc_out_center_w=enc_out_w//2
    crop_w=enc_out_center_w-prev_out_center_w

    encoder_output=encoder_output[:,:,crop_h:enc_out_h-crop_h,crop_w:enc_out_w-crop_w]
    
    return torch.cat((encoder_output,prev_layer_output),dim=1)


class handwritten_cross_entropy(nn.Module):
    def __init__(self,eps=1e-7):
        super(handwritten_cross_entropy,self).__init__()
        self.eps=eps
    def forward(self,logits:torch.Tensor,targets:torch.Tensor):

        logits_max=torch.max(logits,dim=1,keepdim=True)[0]
        # print(logits_max.shape)
        #Normalized logits
        normalized_logits=logits-logits_max
        # print(normalized_logits.shape)
        exp_logits=torch.exp(normalized_logits)

        channel_wise_summed_exp=torch.sum(exp_logits,dim=1,keepdim=True)
        
        probs=exp_logits/channel_wise_summed_exp

        probs=torch.gather(probs,dim=1,index=targets)
        # print(probs.shape)
        log_probs=torch.log(probs+self.eps)
        nll=-torch.mean(log_probs)
        return nll
    

if __name__=="__main__":

    inp=torch.randn(4,2,388,388)
    targets=torch.randint(low=0,high=1,size=(4,1,388,388))
    out=handwritten_cross_entropy()(inp,targets)
    print(out)