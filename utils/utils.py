import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import matplotlib.pyplot as plt
import cv2

def visualize_images(image,nrow):
    grid = torchvision.utils.make_grid(image, nrow=nrow)
    plt.figure(figsize=(50,50))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

def Img2Tensor(img, device):
    img = torch.from_numpy(img).float().to(device).permute(2, 0, 1).unsqueeze(0)
    return img

def visualize_filters(model, dataset, device):
    activation = {}
    def get_activation(name):
        def hook(steering_model, input, output):
            activation[name] = output.detach()
        return hook

    model.model[0].register_forward_hook(get_activation('conv1'))
    data, _ = dataset[0]
    data.unsqueeze_(0)
    output = model(data.to(device).float())

    act = activation['conv1'].squeeze()
    fig, axarr = plt.subplots(act.size(0))
    for idx in range(act.size(0)):
        axarr[idx].imshow(act[idx].cpu())

    plt.show()