from utils import VGGNet, ResNeXt, ConvNeXt, UNetPP, FPN, DeepLab


def cls_model_sel(model, num_class):
    model_type = {
        'VGGNet': VGGNet,
        'ResNeXt': ResNeXt,
        'ConvNeXt': ConvNeXt,
    }
    constructor = model_type.get(model)
    model = constructor(num_class)

    return model


def seg_model_sel(model):
    model_type = {
        'UNetPP': UNetPP,
        'FPN': FPN,
        'DeepLab': DeepLab,
    }
    constructor = model_type.get(model)
    model = constructor()

    return model
