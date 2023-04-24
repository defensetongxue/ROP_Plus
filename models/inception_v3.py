from torchvision import models
import os 
import torch.nn as nn
def build_inception3_pretrained(num_classes,downLoaded='./save_models',
            pretrained=True,print_opt_dic=False):
    os.environ['TORCH_HOME']=downLoaded
    model=models.inception_v3(pretrained=pretrained)
    model.fc=nn.Linear(2048,num_classes)
    model.AuxLogits.fc=nn.Linear(768,num_classes)

    if print_opt_dic:
        opt_dic=model.state_dict().items()
        for k,v in opt_dic:
            print(k," ",v.shape)
    return model
    