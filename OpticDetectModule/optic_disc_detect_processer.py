# this file will create an interface for the rop_dig
from .models import HRNet 
import torch
from PIL import ImageEnhance
import torchvision.transforms as transforms
import math
import numpy as np
import yaml

class OpticDetProcesser():
    def __init__(self):
        with open('./', 'r') as file:
            config = yaml.safe_load(file)
        self.model = HRNet(config)
        checkpoint = torch.load(
            './OpticDetectModule/checkpoint/best.pth')
        self.model.load_state_dict(checkpoint)
        self.model.cuda()

        # generate mask
        self.transforms = transforms.Compose([
            ImageEnhance.Contrast(),
            transforms.Resize((415,416)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4623, 0.3856, 0.2822],
                 std=[0.2527, 0.1889, 0.1334])
            # the mean and std is cal by 12 rop1 samples
            # TODO using more precise score
        ])

    def __call__(self, img,save_path=None):
        # open the image and preprocess
        ori_w,ori_h=img.size
        print(ori_w,ori_h)
        w_ratio,h_ratio=ori_w/416,ori_h/416
        img = self.transforms(img)

        # generate predic heatmap with pretrained model
        img = img.unsqueeze(0)  # as batch size 1
        output = self.model(img.cuda())
        # the input of the 512 is to match the mini-size of vessel model
        score_map = output.data.cpu()
        preds = decode_preds(score_map)
        preds=preds.squeeze()
        print(preds.shape)
        preds=preds*np.array([w_ratio,h_ratio])

        if save_path:
            with open(save_path,'w') as f:
                f.write(f"{int(preds[0])} {int(preds[1])}\n")
        return preds

def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def decode_preds(output):
    map_width,map_height=output.shape[-2],output.shape[-1]
    coords = get_preds(output)  # float type
    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < map_width) and (py > 1) and (py < map_height):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() *.25
    preds = coords.clone()

    # Transform back
    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds*4 # heatmap is 1/4 to original image
