# implement by chatGPT
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms

# Load the Inception_v3 model
model = models.inception_v3(pretrained=True)

# Define the input image size
img_size = (299, 299)

# Define the preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the last convolutional layer
last_conv_layer = model.Mixed_7c

# Define the hook to access the gradients
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output.detach().clone()
    def remove(self): self.hook.remove()

# Define the Grad-CAM function
def generate_grad_cam(model, img_path, last_conv_layer, preprocess, img_size, target_class_idx):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img).unsqueeze(0)
    
    # Register the hook to access the gradients
    features_b = SaveFeatures(last_conv_layer)
    logits = model(img_tensor)
    one_hot_output = torch.zeros_like(logits)
    one_hot_output[0][target_class_idx] = 1
    logits.backward(gradient=one_hot_output)
    
    # Compute the guided gradients
    grads = features_b.features.grad
    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1,1))
    guided_grads = (features_b.features > 0).float() * pooled_grads
    
    # Compute the CAM
    conv_output = features_b.features
    weights = guided_grads.mean((0,2,3)).unsqueeze(1).unsqueeze(1)
    cam = (weights * conv_output).sum(0, keepdim=True)
    
    # Normalize the CAM
    cam = torch.clamp(cam, min=0)
    cam = cam / cam.max()
    
    # Convert the CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam.squeeze().cpu()), cv2.COLORMAP_JET)
    
    # Superimpose the heatmap on the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    heatmap = cv2.resize(heatmap, img_size)
    superimposed_img = heatmap * 0.4 + img
    
    return superimposed_img

# Generate Grad-CAM for an example image
img_path = 'example_image.jpg'
target_class_idx = 0  # class index for the target class
grad_cam = generate_grad_cam(model, img_path, last_conv_layer, preprocess, img_size, target_class_idx)

# Visualize the Grad-CAM
plt.imshow(grad_cam[:,:,::-1])
plt.show()
