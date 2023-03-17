# implement by chatGPT
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
# Load the Inception_v3 model
import torch.nn.functional as F
# Load the Inception_v3 model

# Define the hook to access the gradients
def get_activations_hook(name):
    def hook(model, input, output):
        setattr(model, name, output)
    return hook


def hook_fn(module, input, output):
    global activations
    print(output.shape)
    activations = output.detach()
def generate_heatmap(model, img_tensor, save_path):
    
    # Register the hook for the last convolutional layer
    model.Mixed_7c.register_forward_hook(hook_fn)

    global activations
    model.eval()
    logits = model(img_tensor)
    target_class = torch.argmax(logits, dim=1)
    # Compute the activations and gradients
    _ = model(img_tensor)  # call model to compute activations
    one_hot = F.one_hot(target_class, num_classes=5).float().cuda()
    one_hot.requires_grad_()
    print(activations.shape)
    cam = torch.matmul(one_hot[..., None, None], activations).squeeze()

    cam = cam.squeeze()
    model.zero_grad()
    cam.backward(torch.ones_like(cam))

    # Compute the Grad-CAM
    grad_cam = model.features.weight.grad.mean(dim=[2, 3], keepdim=True)
    cam = F.relu(torch.sum(grad_cam * model.features.weight, dim=1))
    cam = F.interpolate(cam, size=(299, 299), mode='bilinear', align_corners=False)

    # Normalize the Grad-CAM
    cam -= torch.min(cam)
    cam /= torch.max(cam)

    cv2.imwrite(
        os.path.join(save_path, "{}_heatmap.png".format(1)),
        np.uint8((cam * 255).astype('uint8')))
    return