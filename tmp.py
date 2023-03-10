from PIL import Image
import torch
import  torchvision.transforms as transforms
import cv2
import os
transform=transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                # transforms.Normalize()
            ])
img_list_r=[]
img_list_g=[]
img_list_b=[]

for file in os.listdir('./rop-1'):
    img=Image.open(os.path.join('./rop-1',file))
    file=transform(img)
    img_list_r.append(file[0])
    img_list_g.append(file[1])
    img_list_b.append(file[2])

imgR=torch.cat(img_list_r,dim=0)
imgG=torch.cat(img_list_g,dim=0)
imgB=torch.cat(img_list_b,dim=0)

print(imgR.mean(),imgR.std())
print(imgG.mean(),imgG.std())
print(imgB.mean(),imgB.std())

