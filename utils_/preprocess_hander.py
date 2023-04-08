
from VesselSegModule import VesselSegProcesser
from torchvision import transforms

class replace_channel():
    def __init__(self,repalced_channel=2) :
        self.VS_processr=VesselSegProcesser(model_name='FR_UNet',
                                            resize=(300,300))
        self.transform=transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
                                     # the mean and std is calculate by rop1 13 samples
            ])
        self.vessel_transform=transforms.Compose([
            transforms.ToTensor(),
        ])
        self.repalced_channel=repalced_channel
    def replace_channel(self,img,vessel):
        img[self.repalced_channel]=vessel
        return img

    def __call__(self,img) :
        vessel=self.VS_processr(img)
        vessel=self.vessel_transform(vessel)
        img=self.transform(img)
        return self.replace_channel(img,vessel)

class original():
    def __init__(self) :
        self.transform=transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
                                     # the mean and std is calculate by rop1 13 samples
            ])

    def __call__(self,img) :
        img=self.transform(img)
        return img
    
class vessel():
    def __init__(self) :
        self.VS_processr=VesselSegProcesser(model_name='FR_UNet',
                                            resize=(300,300))
        self.transform=transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4623,0.3856,0.2822],
                                     std=[0.2527,0.1889,0.1334])
                                     # the mean and std is calculate by rop1 13 samples
            ])
        self.vessel_transform=transforms.Compose([
            transforms.ToTensor(),
        ])

    def __call__(self,img) :
        vessel=self.VS_processr(img)
        vessel=self.vessel_transform(vessel)
        return vessel.repeat(3,1,1)
