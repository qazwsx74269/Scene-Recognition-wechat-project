# -*- coding: utf-8 -*- 
import time

import torch
from torch.autograd import Variable
from torch import nn,optim
import torchvision.transforms as transforms
import json
import os
import pickle
import PIL
import numpy as np
from functools import partial
from time import strftime
from basic_module import BasicModule

imagepath = "/home1/szh/scene-baseline-master/checkpoints/a.jpg"
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[1.229, 0.224, 0.225])
transformsfunc =  transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])									 
def InputImage(imagepath):
	im = PIL.Image.open(imagepath).convert('RGB')
	return transformsfunc(im)

class ResNet(BasicModule):
    def __init__(self, model,opt=None, feature_dim=2048, name='resnet'):
        super(ResNet, self).__init__(opt)
        self.model_name = name
		# model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		# del model.classif
		# model.classif = lambda x: x
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        del model.fc
        model.fc = lambda x: x
        self.features = model
        self.classifier = nn.Linear(feature_dim, 80)

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

model = torch.load('/home1/szh/scene-baseline-master/checkpoints/whole_resnet50_places365.pth.tar', map_location=lambda storage, loc: storage, pickle_module=pickle)
resnet = ResNet(model)
resnet.load_state_dict(torch.load("/home1/szh/scene-baseline-master/checkpoints/res_365_1203_0055_0.988225053566")['d'])
print("model loaded...")
print(time.asctime(time.localtime(time.time())))
inputimg = Variable(InputImage(imagepath))
inputim = torch.unsqueeze(inputimg,0)
output = resnet(inputim)
_,predicted = torch.max(output.data,1)
print(predicted)
print(time.asctime(time.localtime(time.time())))
print(u"这就是你要的答案")
