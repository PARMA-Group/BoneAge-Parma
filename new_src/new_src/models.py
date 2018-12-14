from model_vgg16 import vgg16
from model_resnet18 import resnet18

def get_model(name, state_dict):
    """
        name can be:
            'vgg16',
            'resnet18'
    """
    model = None
    if name == "vgg16":
        model = vgg16#vgg16(state_dict)
    elif name == "resnet18":
        model = resnet18

    return model(state_dict)