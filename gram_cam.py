# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import cv2
# # in new version of code, this file is useless
# def generate_Gram_CAM(model,target_layer,input_tensor,img_numpy,save_name):
#     cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
#     # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  
#     grayscale_cam = cam(input_tensor=input_tensor)  
#     grayscale_cam = grayscale_cam[0,:]
#     img_numpy=img_numpy[0,:]
#     cam_image = show_cam_on_image(img_numpy, grayscale_cam, use_rgb=True,image_weight=0.8)
#     cv2.imwrite('./Grad_save/{}.jpg'.format(save_name), cam_image)
    

# if __name__=="__main__":
#     from models import build_inception3_pretrained as build_model
#     from utils_ import ImageFolder_Grad
#     import torch
#     from torch.utils.data import DataLoader
#     import numpy as np
#     max_print_success=10
#     max_print_false=10
#     model=build_model(5).cuda()
#     model.load_state_dict(torch.load('./save_models/FR_UNET_blue_channel.pth'))
#     model.eval()
#     datasets=ImageFolder_Grad('../autodl-tmp/')
#     dataloader=DataLoader(datasets,batch_size=1,shuffle=False)
#     print("there is {} data in test".format(len(datasets)))
#     cnt_s=0
#     cnt_f=0
#     for x,img,y in dataloader:
#         logits=model(x.cuda()).cpu()
#         predic=torch.max(logits, 1)[1]
#         predic=str(int(predic[0]))
#         labels=str(int(y[0]))
#         save_name="T{}P{}".format(labels,predic)
#         if predic!= labels and cnt_f<=max_print_false:
#             save_name="F/"+save_name+"_"+str(cnt_f)
#             generate_Gram_CAM(model=model,
#                               target_layer= [model.Mixed_7c],
#                               input_tensor=x,
#                               img_numpy=np.array(img) ,
#                               save_name=save_name)
#             cnt_f+=1
#         elif predic==labels and cnt_s<=max_print_success:
#             save_name="T/"+save_name+"_"+str(cnt_s)
#             cnt_s+=1
#             generate_Gram_CAM(model=model,
#                               target_layer= [model.Mixed_7c],
#                               input_tensor=x,
#                               img_numpy=np.array(img) ,
#                               save_name=save_name)
#         if cnt_s>max_print_success and cnt_f>max_print_false:
#             break