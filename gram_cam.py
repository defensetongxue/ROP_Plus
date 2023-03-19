from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
def generate_Gram_CAM(model,target_layer,input_tensor,img_numpy,targets):
    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  
    grayscale_cam = cam(input_tensor=input_tensor)  
    grayscale_cam = grayscale_cam[0,:]
    img_numpy=img_numpy[0,:]
    cam_image = show_cam_on_image(img_numpy, grayscale_cam, use_rgb=True,image_weight=0.8)
    cv2.imwrite('./res.jpg', cam_image)
    

if __name__=="__main__":
    from CNNs import build_inception3_pretrained as build_model
    from utils import ImageFolder_Grad
    import torch
    from torch.utils.data import DataLoader
    import numpy as np

    model=build_model(5).cuda()
    model.load_state_dict(torch.load('./save_models/FR_UNET_blue_channel.pth'))
    model.eval()
    datasets=ImageFolder_Grad('../autodl-tmp/mini_dataset')
    dataloader=DataLoader(datasets,batch_size=1,shuffle=False)
    for x,img,y in dataloader:
        generate_Gram_CAM(model=model,
                          target_layer= [model.Mixed_7c],
                          input_tensor=x,
                          img_numpy=np.array(img) ,
                          targets=y)
        raise
        