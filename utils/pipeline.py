import os
from PIL import Image
import torch.utils.data as tud
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


def cls_data_aug(dataset, index, bs, device):

    sets = ["augmix", "randaugment", "autoaugment", "trivialaugment", "medaugment"]
    set_type = sets[index-1]
    transform = transforms.ToTensor()

    train_d = ImageFolder(f"./datasets/classification/{dataset}/{set_type}/training", transform=transform)
    val_d = ImageFolder(f"./datasets/classification/{dataset}/{set_type}/validation", transform=transform)
    test_d = ImageFolder(f"./datasets/classification/{dataset}/{set_type}/test", transform=transform)

    if str(device) == 'cuda':
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True)
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True)
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True)

    return train_l, val_l, test_l, train_d, val_d, test_d


class Seg_Dataset(tud.Dataset):
    def __init__(self, path, transform):
        super(Seg_Dataset, self).__init__()
        self.img_path = path
        self.mask_path = path + "_mask"
        self.transform = transform
        self.img_name = [i for i in os.listdir(self.img_path) if i.endswith(".png") or i.endswith(".jpg")]

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.img_path, self.img_name[item]))
        mask_file = os.path.join(self.mask_path, self.img_name[item].split(".")[0] + "_mask")
        if os.path.exists(mask_file + ".png"):
            mask = Image.open(mask_file + ".png")
        elif os.path.exists(mask_file + ".jpg"):
            mask = Image.open(mask_file + ".jpg")

        if not self.transform is None:
            img_rgb, mask_rgb = img.convert('RGB'), mask.convert('RGB')
            img, mask = self.transform(img_rgb), self.transform(mask_rgb)

        else:
            img, mask = img, mask
        return self.img_name[item], img, mask[0,:,:].unsqueeze(0)

    def __len__(self):
        return len(self.img_name)


def seg_data_aug(dataset, index, bs, device):

    sets = ["oneaugment", "twoaugment", "threeaugment", "medaugment"]
    set_type = sets[index-1]
    transform = transforms.ToTensor()

    train_d = Seg_Dataset(f"./datasets/segmentation/{dataset}/{set_type}/training", transform=transform)
    val_d = Seg_Dataset(f"./datasets/segmentation/{dataset}/{set_type}/validation", transform=transform)
    test_d = Seg_Dataset(f"./datasets/segmentation/{dataset}/{set_type}/test", transform=transform)

    if str(device) == 'cuda':
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
    else:
        train_l = tud.DataLoader(train_d, batch_size=bs, shuffle=True)
        val_l = tud.DataLoader(val_d, batch_size=bs, shuffle=True)
        test_l = tud.DataLoader(test_d, batch_size=bs, shuffle=True)

    return train_l, val_l, test_l, train_d, val_d, test_d
