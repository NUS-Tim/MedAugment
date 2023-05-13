import torch


def equipment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print("Warning: cuda unavailable, switch to %s instead" % device)

    return device
