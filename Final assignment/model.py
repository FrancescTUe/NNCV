import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, pretrained=False):
        super().__init__()
        # Load a pretrained DeepLabV3+ with a ResNet-50 backbone
        weights = 'DEFAULT' if pretrained else None
        self.model = deeplabv3_resnet50(
            weights= weights,
            progress=True,
            aux_loss=True
        )

        for param in self.model.parameters():
            param.requires_grad = False

        unfreeze_started = False
        for name, child in self.model.backbone.named_children():
            if name == "layer3": 
                unfreeze_started = True
            
            if unfreeze_started:
                for param in child.parameters():
                    param.requires_grad = True


        for param in self.model.classifier.parameters():
            param.requires_grad = True
        for param in self.model.aux_classifier.parameters():
            param.requires_grad = True

        # Adjust the classifier head for 19 classes
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)
        # Adjust the auxiliary classifier if needed
        self.model.aux_classifier[4] = nn.Conv2d(256, n_classes, kernel_size=1)

    def forward(self, x):
        # DeepLabV3+ returns a dict with 'out' and 'aux'
        return self.model(x)['out']

class VelocityNet(nn.Module):
    def __init__(self, input_dim=3072):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, 1024), # +1 for time 't'
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, input_dim)
        )

    def forward(self, t, x):
        # x is the feature vector, t is the time step [0, 1]
        if t.dim() == 1:
            t = t.unsqueeze(1)

        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)

        tx = torch.cat([x, t], dim=1)
        return self.net(tx)

class FM_OODModel(nn.Module):
    def __init__(self, baseline_model):
        super().__init__()
        self.encoder = baseline_model.model.backbone
        self.classifier = baseline_model.model.classifier
        
        # we freeze the encoder  
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.flow_head = VelocityNet(input_dim=3072) 

    def get_combined_features(self, x):
        """ Helper to extract and combine layer3 and final features correctly """
        # Initial ResNet blocks: conv1 -> bn1 -> relu -> maxpool
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        # ResNet Layers
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        
        # Layer 3 (1024 channels) - Intermediate features
        feat_l3 = self.encoder.layer3(x) 
        
        # Layer 4 (2048 channels) - Final features
        feat_l4 = self.encoder.layer4(feat_l3)
        
        # Global Average Pool both to 1D vectors
        v3 = torch.mean(feat_l3, dim=(2, 3)) # [Batch, 1024]
        v4 = torch.mean(feat_l4, dim=(2, 3)) # [Batch, 2048]
        latent = torch.cat([v3, v4], dim=1)

        latent = F.normalize(latent, p=2, dim=1)
        return latent, feat_l4

    def forward(self, x, return_ood_score=False):
        # we obtain the features and segmentation from baseline model
        latent, features_final = self.get_combined_features(x)
        seg_output = self.classifier(features_final)

        if not return_ood_score:
            return seg_output
        
        ood_score = self.compute_log_likelihood(latent)
        
        return seg_output, ood_score
    
    def compute_log_likelihood(self, z, steps=5):
            total_error = 0
            for i in range(steps):
                t = torch.rand(z.shape[0], 1, device=z.device)
                x0 = torch.randn_like(z)
                xt = (1-t)*x0 + t*z  # Probability path
                target_v = z-x0
                
                pred_v = self.flow_head(t, xt)
                total_error += torch.norm(pred_v-target_v, p=2, dim=1)
                
            return total_error / steps


class StudentModel(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        # MobileNetV3 --> Efficiency benchmark
	
        self.model = deeplabv3_mobilenet_v3_large(
            weights=None, 
          num_classes=n_classes)

    def forward(self, x):
        return self.model(x)['out']


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
