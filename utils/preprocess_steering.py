import torch
import torchvision
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class SteeringDataset(Dataset):

    def __init__(self, root, crop, transform=None):
        super(SteeringDataset, self).__init__()
        self.crop = crop
        self.transform = transform
        # self.image = glob.glob(root+"data/*.jpg")
        self.image = glob.glob(root+"*.jpg")
        self.image = []
        self.angle = []

        with open(root+"data.txt") as f:
            for line in f:
                self.image.append(root + line.split()[0])
                # self.angle.append(float(line.strip().split()[1][:-11])* 3.14159265 / 180.0)
                self.angle.append(float(line.split()[1]))#*3.14159265/180.0)

        self.angle_n = [(((i-min(self.angle))/(max(self.angle)-min(self.angle)))-0.5)/0.5 for i in self.angle] #normalzie all radians values between [-1, 1]
        print(min(self.angle), max(self.angle))

    def get_min_max(self):
        return min(self.angle), max(self.angle)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # working in RGB mode
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image[idx]),cv2.COLOR_BGR2RGB),(200,66))[self.crop:,:] #following w/ nvidia paper implementation
        img = img/255.0
        angle = self.angle_n[idx]
        # if torch.rand(1) > 0.5:
        #     angle = -angle
        #     img = cv2.flip(img,1)
        if self.transform:
            img = self.transform(img)
        return img, angle