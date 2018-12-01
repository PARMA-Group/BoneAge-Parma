import torch.optim as optim

def get_optimizer(model, name, params):
    """
        name can be:
            'Adam',
            'SparseAdam',
            'Adamax',
            'RMSprop',
    """
    optimizer = None
    if name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr = params["lr"], betas = eval(params["betas"]), eps = params["eps"], weight_decay = params["weight_decay"])

    elif name == "SparseAdam":
        optimizer = optim.SparseAdam(model.parameters(), lr = params["lr"], betas = eval(params["betas"]), eps = params["eps"])

    elif name == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=params["lr"], betas=params["betas"], eps=params["eps"], weight_decay=params["weight_decay"])

    elif name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"], weight_decay=params["weight_decay"])

    return optimizer        