import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from BoneDataset import BoneDataset224
import VGG16
import argparse
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        mse = F.mse_loss(x, y)
        loss = torch.sqrt(mse)
        return loss

RMSE_Loss = RMSELoss()

import sys

def main():

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    PATH_TO_IMAGES = "../datasets/dnlm_train/"
    TRAIN_DATASET_CSV = "../utils/male_train.csv" #sys.argv[2]#"/home/icalvo/Proyects/BoneAge/utils/male_train.csv"
    TEST_DATASET_CSV = "../utils/male_test.csv" #sys.argv[3]#"/home/icalvo/Proyects/BoneAge/utils/male_test.csv"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224,224), interpolation=2),
        transforms.ToTensor()
    ])

    train_dataset = BoneDataset224(TRAIN_DATASET_CSV , PATH_TO_IMAGES, transform)
    test_dataset = BoneDataset224(TEST_DATASET_CSV, PATH_TO_IMAGES, transform)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=10,
                                                    shuffle=True)


    PATH_TO_WEIGHTS = "../weights/torchvisionvgg16.pt"
    model = VGG16.vgg16(PATH_TO_WEIGHTS).to(device)
    
    cont = 0
    test_loss = 0
    rmse_loss = 0
    for (data, target) in test_loader:
        with torch.no_grad():
            data, target = data.to(device).float(), target.to(device).float()
            print("batch {}".format(cont))

            output = model(data)

            # loss and
            output = Variable(output, requires_grad=False)
            target = Variable(target.view(target.shape[0],1), requires_grad=False)
            test_loss += F.l1_loss(output, target).item()# sum up batch loss
            rmse_loss += RMSE_Loss(output, target).item()
            cont += 1

            print("output")
            print(output)
            print("\n")
            print("target")
            print(target)
            print("\n" * 3)

    test_loss /= cont #len(test_loader.dataset)
    rmse_loss /= cont #len(test_loader.dataset)
    print("L1 loss: {}".format(test_loss))
    print("RMSE loss: {}".format(rmse_loss))
    
if __name__ == '__main__':
    main()