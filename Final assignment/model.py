import torch
import torch.nn as nn
import timm

class Model(nn.Module):
    """
    High-Resolution Net (HRNet)
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=19,
        pretrained=True
    ):
        """
        Args:
            in_channels (int): Number of input channels
            n_classes (int): Number of output classes (19 for Cityscapes)
        """
        super().__init__()
        self.in_channels = in_channels

        # Load HRNet-W18 backbone
        self.backbone = timm.create_model(
            'hrnet_w18', 
            pretrained=pretrained, 
            features_only=True, 
            in_chans=in_channels
        )

        # HRNet features_only=True returns a list of feature maps from different resolutions.
        # For HRNet-W18, the channels for the 4 parallel streams are 18, 36, 72, 144.
        self.seg_head = nn.Sequential(
            nn.Conv2d(18+36+72+144, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Forward pass with multi-resolution feature fusion.
        """
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")

        # Get features from the 4 parallel resolution streams
        features = self.backbone(x) # [x_1/4, x_1/8, x_1/16, x_1/32]
        
        # Upsample all features to the highest resolution (1/4 of input)
        target_size = features[0].shape[2:]
        upsampled_features = [features[0]]
        for i in range(1, len(features)):
            upsampled_features.append(
                torch.nn.functional.interpolate(features[i], size=target_size, mode='bilinear', align_corners=True))

        # Concatenate features along the channel dimension
        out = torch.cat(upsampled_features, dim=1)
        
        # Apply segmentation head
        logits = self.seg_head(out)

        # Final upsample to match original image input size
        logits = torch.nn.functional.interpolate(
            logits, size=x.shape[2:], mode='bilinear', align_corners=True
        )

        return logits



class U_Net_Model(nn.Module):
    """ 
    A simple U-Net architecture for image segmentation.
    Based on the U-Net architecture from the original paper:
    Olaf Ronneberger et al. (2015), "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    https://arxiv.org/pdf/1505.04597.pdf

    Adapt this model as needed for your problem-specific requirements. You can make multiple model classes and compare them,
    however, the CodaLab server requires the model class to be named "Model". Also, it will use the default values of the constructor
    to create the model, so make sure to set the default values of the constructor to the ones you want to use for your submission.
    """
    def __init__(
        self, 
        in_channels=3, 
        n_classes=19
    ):
        """
        Args:
            in_channels (int): Number of input channels. Default is 3 for RGB images.
            n_classes (int): Number of output classes. Default is 19 for the Cityscapes dataset.
        """
        
        super().__init__()

        # Encoding path
        self.in_channels = in_channels
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 512))

        # Decoding path
        self.up1 = (Up(1024, 256))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        """
        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        
        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits
        

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)