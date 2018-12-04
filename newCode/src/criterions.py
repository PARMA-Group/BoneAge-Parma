import torch.nn as nn
import torch.optim as optim

def get_criterion(name):
    """
        name can be:
            'ce'
    """
    criterion = None
    if name == "ce":
        criterion = nn.CrossEntropyLoss()
    return criterion