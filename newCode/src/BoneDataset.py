import pandas as pd
import numpy as np
import cv2

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class BoneDataset(Dataset):
    def __init__(self, csv_path ,img_path, transform, region):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Folder where images are
        self.img_path = img_path
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.transform = transform
        # Calculate len
        self.data_len = len(self.data_info.index)
        # Whole hand -> A, carpal bones -> B, metacarpal and proximal phalanges -> C.
        self.region = region

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.img_path + str(self.image_arr[index]) + ".png"
        # Open image
        img_as_img = cv2.imread(single_image_name, 0)
        # get region
        if self.region == "B":
            img_as_img = self.get_carpal_region(img_as_img)
        elif self.region == "C":
            img_as_img = self.get_metacarpal_region(img_as_img)

        # Transforms the image
        img_as_tensor = self.transform(img_as_img)

        # Get label
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

    def get_carpal_region(self, img):
        #hardcorded coordinates -> 900x600
        return img[1336:1936, 365:1265]
    def get_metacarpal_region(self, img):
        #hardcorded coordinates -> 1600x600
        return img[656:1256, 0:]