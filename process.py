import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
 
data_transfrom = transforms.Compose([  # 图片读取样式
    transforms.Resize((300, 300)),     
    transforms.ToTensor(),             # 向量化,向量化时 每个点的像素值会除以255,整个向量中的元素值都在0-1之间      
])
 
img = datasets.ImageFolder('../autodl-tmp/rop_1', transform=data_transfrom)  # 指明读取的文件夹和读取方式,注意指明的是到文件夹的路径,不是到图片的路径
 
imgLoader = torch.utils.data.DataLoader(img, batch_size=2, shuffle=False, num_workers=1)  # 指定读取配置信息
 
inputs, _ = next(iter(imgLoader))
print(_[0].item())           # 打印返回的值
inputs = inputs / 2 + 0.5    # 使得整个图像的像素值都在0.5之上，使得处理后的图像偏白
inputs = torchvision.utils.make_grid(inputs)   # make_grid()实现图片的拼接，并去除原本Tesor中Batch_Size那一维度,因为操作之前的inputs是4维的, make_grid()返回的结果是3维的, shape为(3, h, w) 3代表通道数, w,h代表拼接后图片的宽高
inputs = inputs.numpy().transpose((1, 2, 0))   # transpose((1, 2, 0)) 是将(3, h, w) 变为 (h ,w, 3), 因为这种格式才是图像存储的标准格式
plt.imshow(inputs)           # 展示,这里会一块展示batch_size张图片,因为它们是一块被读出来的
plt.show()