import torch.utils.data as data
from PIL import Image,ImageEnhance  
import os
import os.path
import torch
from torchvision import transforms
import json
class ROP_Dataset(data.Dataset):
    '''
        └───data
            │
            └───images
            │   │
            │   └───001.jpg
            │   └───002.jpg
            │   └───...
            │
            └───annotations
                │
                └───train.json
                └───valid.json
                └───test.json
    '''
    def __init__(self, data_path,split='train'):

        
        self.annotations = json.load(open(os.path.join(data_path, 
                                                       'annotations', f"{split}.json")))
        if split=="train":
            self.img_transform=transforms.Compose([
                ImageEnhance.Contrast(factor=1.5),
                transforms.Resize((300,300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                Fix_RandomRotation(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
                # the mean and std is calculate by rop1 13 samples
                ])
        elif split=='val' and split=='test':
            self.img_transform=transforms.Compose([
                ImageEnhance.Contrast(factor=1.5),
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
            ])
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        '''
        The json format is
        {
            "id": <image id> : number,
            "image_name": <image_name> : str,
            "image_name_original": <image_name original> : str,
            "image_path": <image_path> : str,
            "image_path_original": <image_path in original dictionary> : str,
            "class": <class> : number
        }
        '''
        # Load the image and label
        annotation = self.annotations[idx]
        image_path= annotation['image_path']
        img=Image.open(image_path)
        label=annotation['class']

        # Transforms the image
        img=self.img_transform(img)
        
        # Store esscencial data for visualization (Gram)
        meta={}
        meta['image_path']=image_path

        return img,label,meta
    
    def get_rare(self, threshold=0.2):
        # Count the number of samples per class
        class_counts = {}
        for annot in self.annotations:
            cls = annot['class']
            if cls not in class_counts:
                class_counts[cls] = 0
            class_counts[cls] += 1
        
        # Calculate the proportions of each class
        total_samples = len(self.annotations)
        class_proportions = {cls: count / total_samples for cls, count in class_counts.items()}
        
        # Find the rare classes whose proportions are less than the threshold
        rare_classes = [cls for cls, proportion in class_proportions.items() if proportion < threshold]
        
        return rare_classes


class Fix_RandomRotation:
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    def get_params(self):
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        img = transforms.functional.rotate(
            img, angle, transforms.functional.InterpolationMode.NEAREST, 
            expand=self.expand, center=self.center)
        return img
    
class ROP_AugmentedDataset(data.Dataset):
    def __init__(self, original_dataset, class_indices, class_augmentation_factors):
        self.original_dataset = original_dataset
        self.class_indices = class_indices
        self.class_augmentation_factors = class_augmentation_factors
        self.augument_transform=transforms.Compose([
            transforms.RandomRotation(60),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
    def __getitem__(self, index):
        class_idx = index % len(self.class_indices)
        sample_idx = index // len(self.class_indices)
        idx = self.class_indices[class_idx][
            sample_idx % len(self.class_indices[class_idx])]
        img, label, meta = self.original_dataset[idx]

        # Apply random rotation and flipping
        img = self.augument_transform(img)

        return img, label, meta

    def __len__(self):
        return sum(len(indices) * factor for indices, factor in zip(self.class_indices, self.class_augmentation_factors))

def get_factor(dataset, cls, target_samples):
    class_count = sum(1 for annot in dataset.annotations if annot['class'] == cls)
    factor = max(target_samples // class_count - 1, 0)
    return factor
