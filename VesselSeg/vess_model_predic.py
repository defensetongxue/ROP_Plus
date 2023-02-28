import torch
import torch.nn as nn
from VesselSeg import VesModel
import numpy as np
import albumentations
import patchify
import os


class VesselSeg_process(nn.Module):
    def __init__(self, patch_height,
                 patch_width,
                 model_name="LadderNet",
                 pretrained=True,
                 pretrain_path="./VesselSeg/save_model/best_model.pth",
                 save_path=""):
        super(VesselSeg_process, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.pretrained = pretrained
        self.pretrain_path = pretrain_path
        self.save_path = save_path
        if pretrained:
            self.model = self.generate_vessel_seg_model(model_name)
            self.model.eval()
        else:
            print("there is not pretrained ves_model in {}".format(
                self.pretrain_path))
            print("generate a ves model ...")
            raise NotImplementedError

    def generate_vessel_seg_model(self, model_name):
        if model_name == 'LadderNet':
            net = VesModel.LadderNet(inplanes=1, num_classes=2,
                                     layers=3, filters=16).to(self.device)
            if self.pretrained:
                net.load_state_dict(torch.load(self.pretrain_path))
            else:
                raise "don't have pretrain model in {}".format(
                    self.pretrain_path)
            return net
    
    def forward(self, img):
        img_w, img_h = img.size
        patches, patch_number_h, patch_number_w = self.preprocess(img)
        preds = self.model(patches.unsqueeze(1).to(self.device))

        preds = self.recompone_overlap(preds[:, 1, :, :].cpu().detach().numpy(),
                                       patch_number_h, patch_number_w,
                                       img_h, img_w)
        return preds

    def preprocess(self, img):
        '''
        receive: img 
            img.shape=(img_height,img_width,channnel=3)
        return : preprocessed patches:
            shape=(patch_number,1(grayscale),patch_h,patch_w)
        '''
        img = np.array(img)
        transforms = albumentations.Compose([
            albumentations.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
            albumentations.Normalize(
                max_pixel_value=255.0, p=1.0),  # not an idea case
            albumentations.ToGray(),
        ])
        # as the albumentation.transforms.ToGray return three channel image
        # which repeat the 'gray' channel for 3 times
        # consider with the space, only use one of them
        img = transforms(image=img)['image'][:, :, 0]
        print(np.max(img))
        raise
        patches_imgs, patch_number_h, patch_number_w = self.patch_embedding(
            img)
        patches_imgs = torch.tensor(patches_imgs)
        return patches_imgs, patch_number_h, patch_number_w

    def patch_embedding(self, full_imgs):
        full_imgs = self.padding(full_imgs)
        img_shape_after_padding = full_imgs.shape
        patches = patchify.patchify(full_imgs,
                                    (self.patch_height, self.patch_width),
                                    step=self.patch_height)

        patches = patches.reshape(-1, self.patch_height, self.patch_width)
        patch_number_h = img_shape_after_padding/self.patch_height
        patch_number_w = img_shape_after_padding/self.patch_width
        return patches, patch_number_h, patch_number_w

    def padding(self, full_imgs):
        '''
        padding 0 one the right and bottom of the imge
        '''
        img_height, img_width = full_imgs.shape
        padding_height = self.patch_height-(img_height % self.patch_height)
        padding_width = self.patch_width-(img_width % self.patch_width)
        full_imgs = np.pad(full_imgs, ((0, padding_height),
                           (0, padding_width)), 'constant')
        return full_imgs

    def recompone_overlap(self, preds, patch_number_h, patch_number_w, img_h, img_w):
        try:
            preds = preds.reshape(patch_number_h, patch_number_w,
                                  self.patch_height, self.patch_width)
        except:
            print("the model has change the shape of the input")
            raise
        res = patchify.unpatchify(preds,
                                  patch_number_h*self.patch_height,
                                  patch_number_w*self.patch_width)
        res = res[:img_h, :img_w]
        return res

    def save_single_channel_img(self, img, img_name):
        target_path = os.path.join(self.save_path, 'vessel', img_name)
        torch.save(img, target_path)
        return
