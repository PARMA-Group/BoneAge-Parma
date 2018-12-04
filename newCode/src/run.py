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
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

def get_time():
    time = str(datetime.datetime.now())
    time = time.split(" ")
    dot = time[1].index(".")
    time[1] = time[1].replace(":","-")
    return time[0]+"_"+time[1][:dot]

def train():
    pass

def test():
    pass

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

    # Model, optimizer and criterion
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config_file["model"], PATH_TO_WEIGHTS).to(device)
    optimizer = get_optimizer(model, config_file["optimizer"], config_file["optimizer_hyperparameters"])
    criterion = get_criterion(config_file["model"])

    for epoch in range(0, configs["epochs"]):
        #train()
        #test()
        #save_results()
        pass

if __name__ == '__main__':
    main()            