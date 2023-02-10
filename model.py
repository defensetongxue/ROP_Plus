from CNNs.MaskRCNN import MaskRCNN
from CNNs.resUnet import ResUnet
import torch.nn as nn
import torch 
import numpy as np
class ROP_dig(nn.Module):
    def __init__(self,config) -> None:
        super(self,ROP_dig).__init__()
        optic_disc_model=MaskRCNN()
        blood_vessels_model=ResUnet() 
        classifier=nn.Linear()
    def cropping(self,x):
        pass
    def forward(self,x):
        pass
        return x
    