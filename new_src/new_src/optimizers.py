import torch.optim as optim

def get_optimizer(model, name, hyper_parameters):
    """
        name can be:
            'Adam',
            'SparseAdam',
            'Adamax',
            'RMSprop',
    """
    optimizer = None
    if name == "adam":
        optimizer = optim.Adam(model.parameters(), lr = hyper_parameters["lr"], betas = eval(hyper_parameters["betas"]), eps = hyper_parameters["eps"], weight_decay = hyper_parameters["weight_decay"])

    elif name == "sparseadam":
        optimizer = optim.SparseAdam(model.parameters(), lr = hyper_parameters["lr"], betas = eval(hyper_parameters["betas"]), eps = hyper_parameters["eps"])

    elif name == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=hyper_parameters["lr"], betas=hyper_parameters["betas"], eps=hyper_parameters["eps"], weight_decay=hyper_parameters["weight_decay"])

    elif name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=hyper_parameters["lr"], momentum=hyper_parameters["momentum"], weight_decay=hyper_parameters["weight_decay"])

    return optimizer        