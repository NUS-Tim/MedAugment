import torch
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import models
import segmentation_models_pytorch as smp


class VGGNet(torch.nn.Module):
    def __init__(self, num_class):
        super(VGGNet, self).__init__()
        self.model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5, inplace=False),
            torch.nn.Linear(4096, num_class),
        )

    def forward(self, x):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        return self.model(x)


class ResNeXt(torch.nn.Module):
    def __init__(self, num_class):
        super(ResNeXt, self).__init__()
        self.model = models.resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V1')
        self.model.fc = torch.nn.Linear(2048, num_class)

    def forward(self, x):
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        return self.model(x)


class ConvNeXt(torch.nn.Module):
    def __init__(self, num_class):
        super(ConvNeXt, self).__init__()
        self.model = models.convnext_tiny(weights='ConvNeXt_Tiny_Weights.IMAGENET1K_V1')
        self.model.classifier = torch.nn.Sequential(
            models.convnext.LayerNorm2d(768, eps=1e-06),
            torch.nn.Flatten(),
            torch.nn.Linear(768, num_class)
        )

    def forward(self, x):
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        return self.model(x)


class UNetPP(torch.nn.Module):
    def __init__(self):
        super(UNetPP, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)


class FPN(torch.nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.model = smp.FPN(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)


class DeepLab(torch.nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.model = smp.DeepLabV3(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        return self.model(x)
