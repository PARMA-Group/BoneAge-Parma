import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class VGG(nn.Module):
    def __init__(self, num_classes=240, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn
        self.classifier = nn.Sequential(
            nn.Linear(384 * 29 * 22, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def vgg_block_kalinin(self, in_channels, out_channels):
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),
            nn.ELU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        return block
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

"""
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
"""

"""
config = {
    'VGG_KALININ': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG_SAUL': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}    
"""

def vgg_block_kalinin(in_channels, out_channels):
    block = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ELU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1),
        nn.ELU(),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=3, stride=2)
    ]
    return block

def vgg_lineal_kalinin(output_size=2304):
    block = [
        # 244992 -> full img
        # 2304 -> resize y con padding de 1 en las 2 conv2d de arriba
        nn.Linear(output_size, 2048),
        nn.ELU(),
        nn.Dropout2d(0.5),
        nn.Linear(2048, 240),
        nn.ELU()
        #nn.Softmax(dim=1)
    ]    
    return nn.Sequential(*block)

def make_features(features=[32,64,128,128,256,384]):
    layers = []
    in_channels = 1
    for f in features:
        layers += vgg_block_kalinin(in_channels, f)
        in_channels = f

    layers += [nn.BatchNorm2d(in_channels)]
    return nn.Sequential(*layers)
    

x = cv2.imread("../testing/datatest/fitted_train/15600.png", 0)
x = cv2.resize(x, (130, 100))

convs = make_features()
lineals = vgg_lineal_kalinin()

x = torch.tensor([[x]], dtype=torch.float)
x = convs(x)
print(x.shape)
x = x.view(x.size(0), -1)
print(x.shape)
x = lineals(x)
print(x.shape)
print(x)
