import torch.nn as nn
import torchvision.models as models


class AlexDeconv(nn.Module):
    def __init__(self):
        super(AlexDeconv, self).__init__()

        self.features = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1),
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2),
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2),
        )

        self.conv2deconv_indices = {
            0: 12, 3: 9, 6: 6, 8: 4, 10: 2
        }

        self.unpool2pool_indices = {
            0: 12, 7: 5, 10: 2
        }

        self.init_weight()

    def init_weight(self):
        alexnet_pretrained = models.alexnet(pretrained=True)
        for idx, layer in enumerate(alexnet_pretrained.features):
            if isinstance(layer, nn.Conv2d):
                self.features[self.conv2deconv_indices[idx]].weight.data = layer.weight.data

    def forward(self, x, layer, activation_idx, pool_locs):
        if layer in self.conv2deconv_indices:
            start_idx = self.conv2deconv_indices[layer]
        else:
            raise ValueError('layer is not a conv feature map')

        for idx in range(start_idx, len(self.features)):
            if isinstance(self.features[idx], nn.MaxUnpool2d):
                x = self.features[idx] \
                    (x, pool_locs[self.unpool2pool_indices[idx]])
            else:
                x = self.features[idx](x)
        return x
