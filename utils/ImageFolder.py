import torch.utils.data as data
from PIL import Image    
import os
import os.path
import torch
import torchvision.transforms as transforms
# fix from https://blog.csdn.net/xuyunyunaixuexi/article/details/100698216
import pickle
class ROP_Dataset(data.Dataset):
    """
    """
    # 初始化，继承参数
    def __init__(self, root,mode='train'):

        self.class_dic=self._find_classes(root)
        self.imgs=self._make_datasets(root)
        self.mode=mode
        self.transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation()
            ])
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path,label=self.imgs[index]
        with open(img_path,'rb') as file:
            img=pickle.load(file)
        if self.mode=='train':
            img=self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)
    def _make_datasets(self,root):
        img_list=[]
        for target in os.listdir(root):
            target_path=os.path.join(root,target)
            for file in os.listdir(target_path):
                img_list.append(
                    (os.path.join(target_path,file),# file path
                     self.class_dic[target]))# ground truth
        return img_list
    def _find_classes(self,dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
        # example class_to_idx :{'0': 0, '1': 1, '2': 2, '3': 3}
    
             
                