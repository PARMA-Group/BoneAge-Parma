import pandas as pd
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

from torchvision import datasets


class BoneDataset224(Dataset):
    def __init__(self, csv_path ,img_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Folder where images are
        self.img_path = img_path
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        print(csv_path)
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.transform = transform
        #self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.img_path + str(self.image_arr[index]) + ".png"
        # Open image
        img_as_img = Image.open(single_image_name)

        # Transforms the image
        img_as_tensor = self.transform(img_as_img)

        # Get label
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len