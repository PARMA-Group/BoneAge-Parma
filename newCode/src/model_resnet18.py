import torch
import torch.nn as nn
import torchvision.models as models
import os
from torchsummary import summary

def resnet18(state_dict):
    pretrain = False#True if not os.path.isfile(state_dict) else False
    model = models.resnet18(pretrained = pretrain)
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
    #model.classifier[0] = nn.Linear(18432, 4096) 
    #model.classifier[6] = nn.Linear(4096, 240)

    #if not pretrain:
    #    print("* SE CARGARON LOS PESOS *")        
    #    model.load_state_dict(torch.load(state_dict))        
    return model


model = resnet18("a")
print(model)
print(model.conv1)