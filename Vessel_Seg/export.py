import torch
from models.LadderNet import LadderNet
from tmp_lib import *
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn


class vessel_seg_model(nn.Module):
    def __init__(self, model_name="LadderNet",
                 pretrained=True,
                 pretrain_path="./Vessel_Seg/save_model/best_model.pth"):
        super(vessel_seg_model, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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

    def forward(self, data,
                test_patch_height,
                test_patch_width, stride_height,
                stride_width):
        '''
        receive a type data set and return an new dataset
        '''
        img_height, img_width = data.shape[2], data.shape[3]
        data_loader = self.preprocess(
            data, test_patch_height, test_patch_width, stride_height, stride_width)
        preds = []
        for patch in data_loader:
            outputs = self.model(patch.to(self.device))
            outputs = outputs[:, 1].data.cpu().numpy()
            preds.append(outputs)
        predictions = np.concatenate(preds, axis=0)
        pred_patches = np.expand_dims(predictions, axis=1)

        pred_imgs = self.recompone_overlap(
            pred_patches, img_height, img_width, stride_height, stride_width)
        pred_imgs = pred_imgs[:, :, 0:img_height, 0:img_width]
        return pred_imgs

    def preprocess(self, img,
                   test_patch_height,
                   test_patch_width,
                   stride_height,
                   stride_width):
        img = np.array(img)
        imgs = rgb2gray(img)
        imgs = dataset_normalized(imgs)
        imgs = clahe_equalized(imgs)
        imgs = adjust_gamma(imgs, 1.2)
        test_imgs = imgs/255
        test_imgs = self.paint_border_overlap(
            test_imgs, test_patch_height, test_patch_width, stride_height, stride_width)
        patches_imgs_test = self.extract_ordered_overlap(
            test_imgs, test_patch_height, test_patch_width, stride_height, stride_width)
        test_set = TestDataset(patches_imgs_test)
        test_loader = DataLoader(
            test_set, batch_size=64, shuffle=False, num_workers=3)
        return test_loader

    def paint_border_overlap(full_imgs,
                             patch_h, patch_w,
                             stride_h, stride_w):
        assert (len(full_imgs.shape) == 4)  # 4D arrays
        # check the channel is 1 or 3
        assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
        img_h = full_imgs.shape[2]  # height of the image
        img_w = full_imgs.shape[3]  # width of the image
        leftover_h = (img_h-patch_h) % stride_h  # leftover on the h dim
        leftover_w = (img_w-patch_w) % stride_w  # leftover on the w dim
        if (leftover_h != 0):  # change dimension of img_h
            print(
                "\nthe side H is not compatible with the selected stride of " + str(stride_h))
            # print("img_h " +str(img_h) + ", patch_h " +str(patch_h) + ", stride_h " +str(stride_h))
            print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
            print("So the H dim will be padded with additional " +
                  str(stride_h - leftover_h) + " pixels")
            tmp_full_imgs = np.zeros(
                (full_imgs.shape[0], full_imgs.shape[1], img_h+(stride_h-leftover_h), img_w))
            tmp_full_imgs[0:full_imgs.shape[0],
                          0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
            full_imgs = tmp_full_imgs
        if (leftover_w != 0):  # change dimension of img_w
            print(
                "the side W is not compatible with the selected stride of " + str(stride_w))
            # print("img_w " +str(img_w) + ", patch_w " +str(patch_w) + ", stride_w " +str(stride_w))
            print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
            print("So the W dim will be padded with additional " +
                  str(stride_w - leftover_w) + " pixels")
            tmp_full_imgs = np.zeros(
                (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w+(stride_w - leftover_w)))
            tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1],
                          0:full_imgs.shape[2], 0:img_w] = full_imgs
            full_imgs = tmp_full_imgs
        print("new padded images shape: " + str(full_imgs.shape))
        return full_imgs

    def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
        assert (len(full_imgs.shape) == 4)  # 4D arrays
        # check the channel is 1 or 3
        assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)
        img_h = full_imgs.shape[2]  # height of the full image
        img_w = full_imgs.shape[3]  # width of the full image
        assert ((img_h-patch_h) % stride_h ==
                0 and (img_w-patch_w) % stride_w == 0)
        # // --> division between integers
        N_patches_img = ((img_h-patch_h)//stride_h+1) * \
            ((img_w-patch_w)//stride_w+1)
        N_patches_tot = N_patches_img*full_imgs.shape[0]
        print("Number of patches on h : " + str(((img_h-patch_h)//stride_h+1)))
        print("Number of patches on w : " + str(((img_w-patch_w)//stride_w+1)))
        print("number of patches per image: " + str(N_patches_img) +
              ", totally for testset: " + str(N_patches_tot))
        patches = np.empty(
            (N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
        iter_tot = 0  # iter over the total number of patches (N_patches)
        for i in range(full_imgs.shape[0]):  # loop over the full images
            for h in range((img_h-patch_h)//stride_h+1):
                for w in range((img_w-patch_w)//stride_w+1):
                    patch = full_imgs[i, :, h*stride_h:(
                        h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w]
                    patches[iter_tot] = patch
                    iter_tot += 1  # total
        return patches  # array with all the full_imgs divided in patches

    def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):

        assert (len(preds.shape) == 4)  # 4D arrays
        assert (preds.shape[1] == 1 or preds.shape[1]
                == 3)  # check the channel is 1 or 3
        patch_h = preds.shape[2]
        patch_w = preds.shape[3]
        N_patches_h = (img_h-patch_h)//stride_h+1
        N_patches_w = (img_w-patch_w)//stride_w+1
        N_patches_img = N_patches_h * N_patches_w
        # print("N_patches_h: " + str(N_patches_h))
        # print("N_patches_w: " + str(N_patches_w))
        # print("N_patches_img: " + str(N_patches_img))
        assert (preds.shape[0] % N_patches_img == 0)
        N_full_imgs = preds.shape[0]//N_patches_img
        print("There are " + str(N_full_imgs) + " images in Testset")
        full_prob = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
        full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))

        k = 0  # iterator over all the patches
        for i in range(N_full_imgs):
            for h in range((img_h-patch_h)//stride_h+1):
                for w in range((img_w-patch_w)//stride_w+1):
                    # Accumulate predicted values
                    full_prob[i, :, h*stride_h:(h*stride_h)+patch_h,
                              w*stride_w:(w*stride_w)+patch_w] += preds[k]
                    # Accumulate the number of predictions
                    full_sum[i, :, h*stride_h:(h*stride_h)+patch_h,
                             w*stride_w:(w*stride_w)+patch_w] += 1
                    k += 1
        assert (k == preds.shape[0])
        assert (np.min(full_sum) >= 1.0)
        final_avg = full_prob/full_sum  # Take the average
        # print(final_avg.shape)
        assert (np.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (np.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        return final_avg
