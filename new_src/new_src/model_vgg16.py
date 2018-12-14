import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

def vgg16(state_dict):
    pretrain = True if not os.path.isfile(state_dict) else False
    model = models.vgg16(pretrained = False)
    
    model.features[0] = nn.Conv2d(1, 64, kernel_size = 3)
    model.classifier[0] = nn.Linear(18432, 4096) 
    model.classifier[6] = nn.Linear(4096, 240)
    model.classifier.add_module("7", nn.Softmax())

    if not pretrain:   
        model.load_state_dict(torch.load(state_dict))      
    return model


a = vgg16("asd.p")
summary(a, (1,224,224))