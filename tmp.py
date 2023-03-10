from PIL import Image
import torch
import  torchvision.transforms as tf
import cv2
import numpy as np
def mask_vessel(vessel):
    assert len(vessel.shape)==2
    mask=Image.open('res.png')
    mask=tf.Resize(vessel.shape)(mask)
    mask=tf.ToTensor()(mask)[0] 
    res=torch.where(mask<0.1,mask,vessel)
    return res