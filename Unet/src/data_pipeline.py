import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import OxfordIIITPet
from dataclasses import dataclass
from torchvision import transforms
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import os
import pickle

@dataclass
class DataConfig:
    data_dir=r"./Data/image-segmentation"
    artifacts='./artifacts/data_pipeline'
    os.makedirs(artifacts,exist_ok=True)

    input_transform=transforms.Compose(
        [
            transform.Resize((572,572),interpolation=transform.InterpolationMode.BILINEAR),
            transform.ToTensor()
        ]
    )
    gt_transform=transforms.Compose(
        [
            transform.Resize((388,388),interpolation=transform.InterpolationMode.NEAREST),
            transform.ToTensor()
        ]
    )

train_dataset=OxfordIIITPet(root=DataConfig.data_dir,
                            download=True,
                            split='trainval',
                            target_types='segmentation',
                            transform=DataConfig.input_transform,
                            target_transform=DataConfig.gt_transform
                            )

test_dataset=OxfordIIITPet(root=DataConfig.data_dir,
                           download=True,
                           split='test',
                           target_types='segmentation',
                           transform=DataConfig.input_transform,
                           target_transform=DataConfig.gt_transform
                           )

print("Train Dataset size: %s, Test Dataset size: %s"%(len(train_dataset),len(test_dataset)))

def get_train_test_loader():
    train_data_loader=DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,num_workers=0)
    test_data_loader=DataLoader(dataset=test_dataset,batch_size=16,shuffle=False,num_workers=0)

    all_classes=set()
    for _,msks in train_data_loader:
        unique=torch.unique(msks*255).long()
        all_classes.update(unique.tolist())

    with open('./artifacts/data_pipeline/classes.pkl',mode='wb') as file:
        pickle.dump(all_classes,file,protocol=pickle.HIGHEST_PROTOCOL)
        
    return train_data_loader,test_data_loader,all_classes

if __name__=="__main__":
    train_loader,test_loader,classes=get_train_test_loader()

    images,mask=next(iter(train_loader))

    print("Image Shape: ",images.shape)
    print("Mask Shape: ",mask.shape)

    number_of_samples=min(images.shape[0],5)

    fig,axes=plt.subplots(number_of_samples,2,figsize=(10,4*number_of_samples))

    for i in range(number_of_samples):
        img=images[i].permute(1,2,0).numpy()
        msk=mask[i].squeeze().numpy()

        axes[i,0].imshow(img)
        axes[i,0].set_title(f'sample image {i}')
        axes[i,0].axis('off')
        
        axes[i,1].imshow(msk)
        axes[i,1].set_title(f'sample image {i}')
        axes[i,1].axis('off')
    # plt.tight_layout()
    plt.show()