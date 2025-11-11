
from torchvision.models import resnet18
import torch.nn as nn
import torch

class VolClsModel(nn.Module):
    def __init__(self, args):
        super(VolClsModel, self).__init__()
        self.args = args
        self.model = resnet18(weights="IMAGENET1K_V1" if args.pretrained else None)
        self.input_layer = nn.Conv2d(
            in_channels=args.vol_input_channels,
            out_channels=3,
            kernel_size=5,
            padding="same"
        )
        in_features = self.model.fc.in_features
        self.mode.fc = nn.Identity()

        self.fc = nn.Linear(in_features, 1)
    
    def forward(self, x):
        print(x.shape)
        features_ = self.input_layer(x)
        print(features_.shape)
        embed = self.model(features_)
        print(embed.shape)
        preds = self.fc(embed)
        print(preds.shape)
        return preds

class Args:
    pretrained=False
    vol_input_channels=36

if __name__ == "__main__":
    input_ = torch.randn(5, 36, 320, 320)
    args = Args()
    model = VolClsModel(args)
    out_ = model(input_)