import torch
from models.LadderNet import LadderNet
from tmp_lib import *
from torchvision import datasets, transforms
import numpy as npp
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations
import patchify 
import torch.nn.functional as F

class vessel_seg_model(nn.Module):
    def __init__(self,patch_height,
                   patch_width,
                   stride_height,
                   stride_width ,
                   model_name="LadderNet",
                 pretrained=True,
                 pretrain_path="./Vessel_Seg/save_model/best_model.pth"):
        super(vessel_seg_model, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.patch_height=patch_height
        self.patch_width=patch_width
        self.stride_height=stride_height
        self.stride_width=stride_width
        assert self.stride_height==self.stride_width

        self.pretrained = pretrained
        self.pretrain_path = pretrain_path

        self.model = self.generate_vessel_seg_model(model_name)

    def generate_vessel_seg_model(self, model_name):
        if model_name == 'LadderNet':
            net = LadderNet(inplanes=1, num_classes=2,
                            layers=3, filters=16).to(self.device)
            if self.pretrained:
                net.load_state_dict(torch.load(self.pretrain_path))
            else:
                raise "don't have pretrain model in {}".format(
                    self.pretrain_path)
            return net

    def forward(self, data):
        '''
        receive a type data set and return an new dataset
        '''
        img_height, img_width = data.shape[2], data.shape[3]
        data_loader = self.preprocess(data)
        preds = []
        for patch in data_loader:
            outputs = self.model(patch.to(self.device))
            outputs = outputs[:, 1].data.cpu().numpy()
            preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        pred_patches = np.expand_dims(predictions, axis=1)

        pred_imgs = self.recompone_overlap(
            pred_patches, img_height, img_width)
        pred_imgs = pred_imgs[:, :, 0:img_height, 0:img_width]
        return pred_imgs

    def preprocess(self, img):
        transforms = albumentations.Compose([
        albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
        max_pixel_value=255.0, p=1.0),
        albumentations.ToGray()
        ])
        img=transforms(img)
        patches_imgs = self.patch_embedding(img)
        patches_imgs = torch.tensor(patches_imgs)
        
        test_set = datasets(patches_imgs)
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=3)
        return test_loader

    
    def patch_embedding(self,full_imgs):
        try:
            img_number,channels,_,_=full_imgs.shape
        except:
            print("only receive 4D channels imgs")
            raise
        full_imgs=self.padding(full_imgs)
        self.img_shape_after_padding=full_imgs.shape
        patches=patchify.patchify(full_imgs,
                                  (img_number,channels,self.patch_height,self.patch_width),
                                  step=self.patch_height)
        self.patch_orignal_shape=patches.shape # used for reconstruct
        patches=patches.reshape(-1,self.patch_height,self.patch_width)
          
        return patches
    def padding(self,full_imgs):
        img_height,img_width = full_imgs.shape[2],full_imgs.shape[3]
        padding_height=(img_height-self.patch_height) % self.stride_height
        padding_width=(img_width-self.patch_width) % self.stride_width
        # F.pad(tensor,(left,right, top,bottom))
        full_imgs=F.pad(full_imgs,(0,padding_height,0,padding_width),'constant',0)

        return full_imgs

    def recompone_overlap(self,preds, img_h, img_w):
        try:
            preds=preds.reshape(self.patch_orignal_shape)
        except:
            print("the model has change the shape of the input")
            raise
        res=patchify.unpatchify(preds,self.img_shape_after_padding)
        res=res[:,:,:img_h,img_w]
        return res