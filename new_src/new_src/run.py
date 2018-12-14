import sys
import json
import os
import datetime
from shutil import copy2

import torch
import torch.nn as nn

from torchvision import transforms
from BoneDataset import BoneDataset
from criterions import get_criterion
from optimizers import get_optimizer
from models import get_model
from Results import Results

# Set cuda float tensor as default type
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# this function is for the results directory name
def get_time():
    time = str(datetime.datetime.now())
    time = time.split(" ")
    dot = time[1].index(".")
    time[1] = time[1].replace(":", "-")
    return time[0]+"_"+time[1][:dot]

def train(model, device, train_loader, optimizer, criterion, epoch, log_interval, results):
    # train mode
    model.train()

    # iterate over the data and targets
    for batch_idx, (data, target) in enumerate(train_loader):
        # set data and target to GPU
        data, target = data.to(device), target.to(device)
        # grads on zero
        optimizer.zero_grad()
        # forward
        output = model(data)
        # computes the loss
        loss = criterion(output, target)
        # computes the gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # print the loss per interval
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # saves the loss in a csv file
    results.add_train_result(loss.item())


def test(model, device, test_loader, criterion, mae, results):
    # test mode
    model.eval()

    # aux variables
    test_loss = 0
    test_mae = 0
    batches = 0

    # torch please don't compute gradients
    with torch.no_grad():
        # iterate over the test data
        for data, target in test_loader:
            # set data and target to GPU
            data, target = data.to(device), target.to(device)
            # forwards
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # sum up correct mean absolute error
            test_mae += mae(output, target).item()
            batches += 1

    # average loss
    test_loss /= batches
    test_mae /= batches
    print('\nTest set: Average loss: {:.4f}, MAE: {} \n'.format(test_loss, test_mae))
    # saves the results in a csv file
    results.add_test_result(test_loss, test_mae)

def main():
    # read configuration file
    with open(sys.argv[1]) as configs:
        config_file = json.load(configs)

    # Load all the paths
    PATH_TO_IMAGES = config_file["path_to_images"]
    TRAIN_DATASET_CSV = config_file["path_to_train_csv"]
    TEST_DATASET_CSV = config_file["path_to_test_csv"]
    PATH_TO_WEIGHTS = config_file["path_to_weights"]
    PATH_TO_RESULTS = config_file["path_to_results"]
    WEIGHTS_FILE = PATH_TO_IMAGES + "weights.pt"

    # Creates the results folder
    # This folder will contain the train and test results, config file and weights of the model
    results_directory = PATH_TO_RESULTS + get_time() + "/"
    os.mkdir(results_directory)
    copy2(sys.argv[1], results_directory)

    # Transform of the images
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)), # Image size
        transforms.ToTensor()
    ])

    # Datasets
    train_dataset = BoneDataset(TRAIN_DATASET_CSV, PATH_TO_IMAGES, transform, config_file["region"])
    test_dataset = BoneDataset(TEST_DATASET_CSV, PATH_TO_IMAGES, transform, config_file["region"])

    # Train loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config_file["train_batch_size"], shuffle=True)

    # Test loader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=config_file["test_batch_size"], shuffle=True)

    # device, model, optimizer, criterion , MAE and results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config_file["model"], PATH_TO_WEIGHTS).to(device)
    optimizer = get_optimizer(model, config_file["optimizer"], config_file["optimizer_hyperparameters"])
    criterion = get_criterion(config_file["criterion"])
    mae = nn.L1Loss()
    results = Results(results_directory)

    for epoch in range(0, configs["epochs"]):
        train(model, device, train_loader, optimizer, criterion, epoch, config_file["log_interval"], results)
        test(model, device, test_loader, criterion, mae, results)
        torch.save(model.state_dict(), WEIGHTS_FILE)
        results.write_results()

if __name__ == '__main__':
    main()