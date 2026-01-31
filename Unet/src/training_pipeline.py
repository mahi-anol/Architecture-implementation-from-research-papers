from data_pipeline import get_train_test_loader
from model import UNET
from torch.optim import Adam
from utils import handwritten_cross_entropy
import torch
import numpy as np
import random
from tqdm import tqdm
import logging
from utils import saving_model_with_state_and_logs
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device='cuda' if torch.cuda.is_available() else 'cpu'

train_loader,test_loader,all_classes=get_train_test_loader()

n_class=len(all_classes)

model=UNET(in_channel=3,out_Channels=n_class)
model.to(device)
optimizer=Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)

loss_fn=handwritten_cross_entropy()



def train_step(model,data_loader,loss_fn,optimizer,device):
    model.train()
    train_loss,train_accuracy=0.0,0.0
    bar=tqdm(data_loader,desc='Training Epoch going on',leave=False)
    for batch,(x,Y) in enumerate(bar):
        x,Y=x.to(device),Y.to(device).long()#y=b,1,h,w
        # print("type of Y is ",type(x))
        logits=model(x) #b,c,h,w
        batch_loss=loss_fn(logits,Y)
        train_loss+=batch_loss.item()
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        y_pred=torch.argmax(torch.softmax(logits,dim=1),dim=1,keepdim=True)#b,1,h,w

        batch_accuracy=(y_pred==Y).sum().item()/Y.numel()
        train_accuracy+=batch_accuracy
        bar.set_postfix(batch_loss=f'{batch_loss}',batch_accuracy=f'{batch_accuracy*100}%')
    train_loss=train_loss/len(data_loader)
    train_accuracy=(train_accuracy/len(data_loader))*100
    return train_accuracy,train_loss

def test_step(model,data_loader,loss_fn,device):
    model.eval()
    test_loss, test_accuracy=0.0,0.0
    with torch.inference_mode():
        bar=tqdm(data_loader,desc='Testing Epoch going on',leave=False)
        for batch,(x,Y) in enumerate(bar):
            x,Y=x.to(device),Y.to(device).long() #y=b,1,h,w
            logits=model(x)
            batch_loss=loss_fn(logits,Y)
            test_loss+=batch_loss.item()
            y_pred=torch.argmax(torch.softmax(logits,dim=1),dim=1,keepdim=True) #b,1,h,w
            batch_accuracy=(y_pred==Y).sum().item()/Y.numel()
            test_accuracy+=batch_accuracy
            bar.set_postfix(batch_loss=f'{batch_loss}',batch_accuracy=f'{batch_accuracy*100}%')
        test_loss=test_loss/len(data_loader)
        test_accuracy=(test_accuracy/len(data_loader))*100
    return test_accuracy,test_loss

def train(model,train_dataloader,test_dataloader,optimizer,loss_fn,epochs,device,checkpoint_saving_gap):

    best_test_accuracy = -float('inf') 
    best_test_loss=float('inf')
    ### storing logs
    results={
        "train_loss":[],
        "test_loss":[],
        "train_accuracy":[],
        "test_accuracy":[],
    }
    for epoch in range(epochs):
        train_accuracy,train_loss=train_step(model,train_dataloader,loss_fn,optimizer,device)
        test_accuracy,test_loss=test_step(model,test_dataloader,loss_fn,device)
        logging.info(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_accuracy:.4f}')
        results['train_accuracy'].append(train_accuracy)
        results['train_loss'].append(train_loss)
        results["test_accuracy"].append(test_accuracy)
        results['test_loss'].append(test_loss)
         # Save checkpoint every nth epoch
        if (epoch + 1) % checkpoint_saving_gap == 0:
            # It's good practice to reflect the loss type in the checkpoint name if it differs,
            # but based on the overall script, this engine is specifically for cross_entropy.
            saving_model_with_state_and_logs(model, optimizer, results, f"{epoch+1}_crossentropy_loss_trained_model.pt")
            logging.info(f"Saved epoch checkpoint at epoch {epoch+1}")
        # Save best model based on test accuracy
        if test_accuracy > best_test_accuracy:
            logging.info("weights from current epoch outperformed previous weights. ")
            best_test_accuracy = test_accuracy
            logging.info(f"Saving best model with Test Accuracy: {best_test_accuracy:.4f} at epoch {epoch+1} @ ./checkpoint")
            # When saving 'best.pt', ensure 'results' reflects the metrics *up to that point*.
            # A shallow copy is usually sufficient if saving_model_with_state_and_logs doesn't modify it.
            # Using slice [:] creates a shallow copy of the lists within results.
            current_results_for_best = {k: v[:] for k, v in results.items()} 
            saving_model_with_state_and_logs(model, optimizer, current_results_for_best, "cross_entropy_best.pt")

    # After the training loop finishes, save the last model
    logging.info("Saving last trained model @ ./models")
    saving_model_with_state_and_logs(model, optimizer, results, "cross_entropy_last.pt")



    return results

# if __name__=="__main__":
train(model=model
    ,train_dataloader=train_loader
    ,test_dataloader=test_loader
    ,optimizer=optimizer
    ,loss_fn=loss_fn
    ,epochs=2
    ,device=device
    ,checkpoint_saving_gap=1
    )