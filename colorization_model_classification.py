import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
# You might not have tqdm, which gives you nice progress bars
from tqdm import tqdm
import copy

import pandas as pd

from skimage import io, color
from skimage import data
from skimage.color import rgb2lab, lab2lch, lab2rgb


class colorization_model_zhang(nn.Module):
	
	def __init__(self,number_lab_classes):
		super(colorization_model_zhang, self).__init__()

		size = [64,128,256,512,512,512,512,256]
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=size[0], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[0], out_channels=size[1], kernel_size=3, stride=2, padding=1,dilation=1),
			nn.ReLU(),
			nn.BatchNorm2d(size[1])
		)

		self.conv2 = nn.Sequential(
			nn.Conv2d(in_channels=size[1], out_channels=size[1], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[1], out_channels=size[2], kernel_size=3, stride=2, padding=1,dilation=1),
			nn.ReLU(),
			nn.BatchNorm2d(size[2])
		)

		self.conv3 = nn.Sequential(
			nn.Conv2d(in_channels=size[2], out_channels=size[2], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[2], out_channels=size[2], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[2], out_channels=size[3], kernel_size=3, stride=2, padding=1,dilation=1),
			nn.ReLU(),
			nn.BatchNorm2d(size[3])
		)

		self.conv4 = nn.Sequential(
			nn.Conv2d(in_channels=size[3], out_channels=size[3], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[3], out_channels=size[3], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[3], out_channels=size[4], kernel_size=3, stride=1, padding=1,dilation=1),
			nn.ReLU(),
			nn.BatchNorm2d(size[4])
		)

		self.conv5 = nn.Sequential(
			nn.Conv2d(in_channels=size[4], out_channels=size[4], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[4], out_channels=size[4], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[4], out_channels=size[5], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.BatchNorm2d(size[5])
		)

		self.conv6 = nn.Sequential(
			nn.Conv2d(in_channels=size[5], out_channels=size[5], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[5], out_channels=size[5], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[5], out_channels=size[6], kernel_size=3, stride=1, padding=2,dilation=2),
			nn.ReLU(),
			nn.BatchNorm2d(size[6]),
			nn.Upsample(scale_factor=2)
		)

		self.conv7 = nn.Sequential(
			nn.Conv2d(in_channels=size[6], out_channels=size[6], kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[6], out_channels=size[7], kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Upsample(scale_factor=4),
			nn.BatchNorm2d(size[7])
		)

		self.conv8 = nn.Sequential(
			nn.Conv2d(in_channels=size[7], out_channels=size[7], kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.Conv2d(in_channels=size[7], out_channels=number_lab_classes, kernel_size=3, stride=1, padding=1),
			nn.Softmax2d()
		)

	def forward(self, x):
		#x.size = [batch_size,1,256,256]
		x = self.conv1(x)
		#x.size = [batch_size,1,128,128]
		#print("conv1",x.shape)
		x = self.conv2(x)
		#x.size = [batch_size,1,64,64]
		#print("conv2",x.shape)
		x = self.conv3(x)
		#x.size = [batch_size,1,32,32]
		#print("conv3",x.shape)
		x = self.conv4(x)
		#x.size = [batch_size,1,122,122]
		#print("conv4",x.shape)
		x = self.conv5(x)
		#x.size = [batch_size,1,122,122]
		#print("conv5",x.shape)

		x = self.conv6(x)
		#x.size = [batch_size,1,122,122]
		#print("conv6",x.shape)
		x = self.conv7(x)
		#x.size = [batch_size,number_lab_classes,244,244]
		#print("conv7",x.shape)
		x = self.conv8(x)
		
		return x