# this file will create an interface for the rop_dig
from . import models
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np


class VesselSegProcesser():
    def __init__(self, model_name,
                 resize=(512, 512),
                 save_img=False,
                 save_path=None):
        self.model = getattr(models, model_name)()
        checkpoint = torch.load(os.path.join(
            './VesselSegModule', 'checkpoint/best.pth'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.cuda()

        self.save_path = save_path
        if save_img and not os.path.exists(save_path):
            os.mkdir(save_path)
        self.resize = resize
        self.save_img = save_img
        # generate mask
        mask = Image.open('./VesselSegModule/mask.png')
        mask = transforms.Resize(self.resize)(mask)
        mask = transforms.ToTensor()(mask)[0]
        self.mask = mask

        self.transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([0.3968], [0.1980])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
        ])

    def __call__(self, img):
        # open the image and preprocess
        # img = Image.open(img_path)
        img = self.transforms(img)

        # generate predic vascular with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        pre = self.model(img.cuda())
        # the input of the 512 is to match the mini-size of vessel model
        pre = transforms.functional.crop(pre, 0, 0, 512,512)
        
        pre=transforms.Resize(self.resize)(pre)
        pre = pre[0, 0, ...]
        predict = torch.sigmoid(pre).cpu().detach()
        # mask
        predict = torch.where(self.mask < 0.1, self.mask, predict)
        if self.save_img:
            #save the image
            # file_name = os.path.basename(img_path)
            file_name = file_name.split('.')[0]
            cv2.imwrite(
                os.path.join(self.save_path, "{}_vs.png".format(file_name)),
                np.uint8(predict.numpy()*255))
        return predict.numpy()
