import torch.utils.data as data
from PIL import Image    
import os
import os.path
import torch
import torchvision.transforms as transforms
# fix from https://blog.csdn.net/xuyunyunaixuexi/article/details/100698216

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
    # example class_to_idx :{'0': 0, '1': 1, '2': 2, '3': 3}
 
 
def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for file_name in sorted(os.listdir(d)):
            item = (os.path.join(d,file_name),file_name.split('.')[0], class_to_idx[target])  
            images.append(item)
 
    return images



def replace_channel(img,vessel,channel):
    img[channel]=vessel
    return img

class ImageFolder_ROP(data.Dataset):
    """
    """
    # 初始化，继承参数
    def __init__(self, root):

        classes, class_to_idx = find_classes(os.path.join(root,'test'))
        imgs = make_dataset(os.path.join(root,'test'), class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root ))
 
        self.root = root
        self.vessel_path=os.path.join(root,'vessel_res')
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform=transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
                                     # the mean and std is calculate by rop1 13 samples
            ])
        self.vessel_transform=transforms.Compose([
            transforms.Resize((300,300)),
            transforms.ToTensor(),
            lambda x:x[0]
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path,file_name, target = self.imgs[index] 
        img = Image.open(path) 
        vessel=Image.open(os.path.join(self.vessel_path,
                                            "{}_vs.png".format(file_name)))
        
        if self.transform is not None:
            img = self.transform(img)
        if self.vessel_transform is not None:
            vessel=self.vessel_transform(vessel)
        img=replace_channel(img,vessel,2)
        return img, target
 
    def __len__(self):
        return len(self.imgs)
    