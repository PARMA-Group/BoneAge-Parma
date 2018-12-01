import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class VGG(nn.Module):
    def __init__(self, num_classes=240, init_weights=True):
        super(VGG, self).__init__()

        in_channels = 1

        self.features = nn.Sequential(
            *self.feature_block(in_channels, 32),
            *self.feature_block(32, 64),
            *self.feature_block(64, 128),
            *self.feature_block(128, 256),
            *self.feature_block(256, 384),
            nn.Dropout()
        )

        self.classifier = nn.Sequential(
            nn.Linear(384 * 29 * 22, 2048),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(2048, 1),
            nn.ELU()
        )
        if init_weights:
            self._initialize_weights()

    def feature_block(self, in_channels, out_channels):
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
        print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)
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