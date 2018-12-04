import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from BoneDataset import BoneDataset224

import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
from LogCSV import LogCSV
import json


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        mse = F.mse_loss(x, y)
        loss = torch.sqrt(mse)
        return loss
        

performance = LogCSV("L1Loss","RMSE")

def train(configs, model, device, train_loader, optimizer, criterion, epoch):
    model.train()    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data).float()
        target = target.view(target.shape[0],1)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
                
        if batch_idx % configs["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(configs, model, device, loss, criterion, test_loader):
    model.eval()
    test_loss = 0
    rmse_loss = 0
    batches = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data).float()
            
            # Esto para que no de error el auto grad
            output = Variable(output, requires_grad=False)
            target = Variable(target.view(target.shape[0],1), requires_grad=False)
            test_loss += criterion(output, target).item()# sum up batch loss
            rmse_loss += loss(output, target).item()
            batches += 1
            
            # guarda el predict en el csv
    
    test_loss /= batches #len(test_loader.dataset)
    rmse_loss /= batches #len(test_loader.dataset)
    performance.add_result(test_loss, rmse_loss)
    print('\nTest set: Average L1loss: {:.4f}, Average RMSEloss: {:.4f}\n'.format(test_loss, rmse_loss))

def main():
    
    configs = {}
    with open(sys.argv[1]) as f:
        configs = json.load(f)
    
    print(configs)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carga de todas las direcciones
    PATH_TO_IMAGES = configs["path_to_images"]#"../datasets/dnlm_train/"
    TRAIN_DATASET_CSV = configs["path_to_train_csv"] #"../utils/male_train.csv" #sys.argv[2]#"/home/icalvo/Proyects/BoneAge/utils/male_train.csv"
    TEST_DATASET_CSV = configs["path_to_test_csv"]#"../utils/male_test.csv" #sys.argv[3]#"/home/icalvo/Proyects/BoneAge/utils/male_test.csv"
    PATH_TO_WEIGHTS = configs["path_to_weights"]

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224,224), interpolation=2),
        transforms.ToTensor()
    ])

    train_dataset = BoneDataset224(TRAIN_DATASET_CSV , PATH_TO_IMAGES, transform)
    test_dataset = BoneDataset224(TEST_DATASET_CSV, PATH_TO_IMAGES, transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=configs["batch_size"],
                                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=configs["test_batch_size"],
                                                    shuffle=True)


     #"../weights/torchvisionvgg16.pt"
    model = VGG16.vgg16(PATH_TO_WEIGHTS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs["lr"])
    criterion = nn.L1Loss()
    RMSE_Loss = RMSELoss()
    
    for epoch in range(1, configs["epochs"] + 1):
        train(configs, model, device, train_loader, optimizer, criterion, epoch)
        test(configs, model, device, RMSE_Loss, criterion, test_loader)
        torch.save(model.state_dict(), PATH_TO_WEIGHTS)
        performance.make_csv(configs["csv_performance_name"])

if __name__ == '__main__':
    main()