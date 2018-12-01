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
import sys
from Plots import Plotlight


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        mse = F.mse_loss(x, y)
        loss = torch.sqrt(mse)
        return loss

RMSE_Loss = RMSELoss()
plotting = Plotlight()
plotting.labels = ["L1 Loss", "RMSE"]
  

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data).float()
        
        # Esto para que no de error el auto grad
        output = Variable(output, requires_grad=True)
        target = Variable(target.view(target.shape[0],1), requires_grad=True)
        
        loss = F.l1_loss(output, target)
        loss.backward()
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    plotting.reset()
    test_loss = 0
    rmse_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data).float()
            
            # Esto para que no de error el auto grad
            output = Variable(output, requires_grad=False)
            target = Variable(target.view(target.shape[0],1), requires_grad=False)
            test_loss += F.l1_loss(output, target).item() # sum up batch loss
            rmse_loss += RMSE_Loss(output, target).item()
            #pred = output #.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    rmse_loss /= len(test_loader.dataset)
    plotting.results.append([test_loss, rmse_loss])
    print('\nTest set: Average L1loss: {:.4f}, Average RMSEloss: {:.4f}\n'.format(test_loss, rmse_loss))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print("VGG16 SGD")
    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    PATH_TO_IMAGES = "../datasets/fitted_train_temp/"
    print(PATH_TO_IMAGES)
    TRAIN_DATASET_CSV = "../utils/male_train.csv" #sys.argv[2]#"/home/icalvo/Proyects/BoneAge/utils/male_train.csv"
    TEST_DATASET_CSV = "../utils/male_test.csv" #sys.argv[3]#"/home/icalvo/Proyects/BoneAge/utils/male_test.csv"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224,224), interpolation=2),
        transforms.ToTensor()
    ])

    train_dataset = BoneDataset224(TRAIN_DATASET_CSV , PATH_TO_IMAGES, transform)
    test_dataset = BoneDataset224(TEST_DATASET_CSV, PATH_TO_IMAGES, transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=32,
                                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=64,
                                                    shuffle=True)


    PATH_TO_WEIGHTS = "../weights/ws_sdg_male_no_dnlnm.pt"
    model = VGG16.vgg16(PATH_TO_WEIGHTS).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        torch.save(model.state_dict(), PATH_TO_WEIGHTS)   
    plotting.plot("vgg16-sgd-male-no-dnlm.png")


if __name__ == '__main__':
    main()