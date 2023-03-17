from PIL import Image
import torch
import  torchvision.transforms as transforms
from utils import generate_heatmap
from CNNs.inception_v3 import build_inception3_pretrained as build_model
model = build_model(5)
model.load_state_dict(
    torch.load('./save_models/FR_UNET_blue_channel.pth'))
# Define the input image size

# Visualize the Grad-CAM
from utils import ImageFolder_ROP
from torch.utils.data import DataLoader
PATH='../autodl-tmp/mini_dataset'
dataset=ImageFolder_ROP(PATH)
data_loader=DataLoader(dataset,1)
cnt=0
for test_data,_ in data_loader:
    generate_heatmap(model,test_data,'./tmp_save_path')
    cnt+=1
    raise