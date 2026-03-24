import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from transformers import SegformerModel

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

    def forward(self, x, return_features=False):
        features = self.model.backbone(x)
        x_features = self.model.classifier[0:4](features['out'])
        logits = self.model.classifier[4](x_features)
        
        if return_features:
            latent_vec = torch.mean(x_features, dim=(2, 3))
            return logits, latent_vec
        
        return logits
    
class TimeSinusoidal(nn.Module):
    def __init__(self, time_embed_dim=64):
        super().__init__()
        self.d_model = time_embed_dim

    def forward(self, t):
        device = t.device
        t_scaled = t*1000 
        # we compute the divergence term
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()*-(torch.log(torch.tensor(10000.0, device=device))/self.d_model)
        )
        
        # we apply the sin/cos to the continuous t
        embeddings = torch.zeros((t.shape[0], self.d_model), device=device)
        embeddings[:, 0::2] = torch.sin(t_scaled*div_term)
        embeddings[:, 1::2] = torch.cos(t_scaled*div_term)
        
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.1) # Prevents overfitting to Cityscapes noise
        )

    def forward(self, x):
        return x + self.block(x) # Skip connection

class VelocityNet(nn.Module):
    def __init__(self, input_dim=3072, time_embed_dim=64):
        super().__init__()
        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        # Small MLP to embed the time scalar t into a higher-dimensional space
        self.sin_time = TimeSinusoidal(time_embed_dim=time_embed_dim)

        self.time_embed = nn.Sequential(
        nn.Linear(time_embed_dim, time_embed_dim),
        nn.SiLU(),                     # Activation function: Sigmoid Linear Unit
        nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Stacked residual blocks for deeper reasoning
        self.res_blocks = nn.Sequential(
            ResidualBlock(1024),
            ResidualBlock(1024)
        )
        
        # Final projection back to feature space
        self.output_proj = nn.Linear(1024, input_dim)

    def forward(self, t, x):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if t.shape[0] != x.shape[0]:
            t = t.expand(x.shape[0], 1)
        t_embed = self.time_embed(self.sin_time(t))

        tx = torch.cat([x, t_embed], dim=1)
        x = self.input_proj(tx)
        x = self.res_blocks(x)
        return self.output_proj(x)

class FM_OODModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")      
        # we freeze the encoder  
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.flow_head = VelocityNet(input_dim=256) 
        #self.flow_head = VelocityNet(input_dim=480) 

    def forward(self, x):
        # we obtain the features from the ViT
        outputs = self.encoder(x, output_hidden_states=True)

        #s2 = torch.mean(outputs.hidden_states[2], dim=[2, 3])
        #s3 = torch.mean(outputs.hidden_states[3], dim=[2, 3])
        #s4 = torch.mean(outputs.hidden_states[4], dim=[2, 3])
        #multi_scale_latent = torch.cat([s2, s3, s4], dim=1)
        
        features = outputs.last_hidden_state
        latent_vector = torch.mean(features, dim=[2,3])
        latent = F.normalize(latent_vector, p=2, dim=1)

        ood_score = self.compute_log_likelihood(latent)
        
        return ood_score
    
    def compute_log_likelihood(self, z, steps=5):
        total_error = 0
        t_steps = torch.linspace(0.1, 1.0, steps, device=z.device)
        for t_val in t_steps:
            t = torch.full((z.shape[0], 1), t_val, device=z.device)
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
