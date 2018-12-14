import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

def inceptionv3(state_dict):
    pretrain = True if not os.path.isfile(state_dict) else False
    model = models.inception_v3(pretrained = False)

    model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    model.aux_logits = False
    
    #if not pretrain:   
    #    model.load_state_dict(torch.load(state_dict))
    return model


model = inceptionv3("a")
#print(model)
summary(model, (1, 299, 299))