import segmentation_models_pytorch as smp
import torch
import logging
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

# List of supported models for 'segmentation_models_pytorch' package
supported_models = {
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
    "manet": smp.MAnet,
    "linknet": smp.Linknet,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "pan": smp.PAN,
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
}

# List of supported encoders for 'segmentation_models_pytorch' package
supported_encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x4d",
    "resnext101_32x8d",
    "resnext101_32x16d",
    "resnext101_32x32d",
    "resnext101_32x48d",
    "dpn68",
    "dpn68b",
    "dpn92",
    "dpn98",
    "dpn107",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "inceptionresnetv2",
    "inceptionv4",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "mobilenet_v2",
    "xception",
    "timm-efficientnet-b0",
    "timm-efficientnet-b1",
    "timm-efficientnet-b2",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b5",
    "timm-efficientnet-b6",
    "timm-efficientnet-b7",
    "timm-efficientnet-b8",
    "timm-efficientnet-l2",
    "timm-tf_efficientnet_lite0",
    "timm-tf_efficientnet_lite1",
    "timm-tf_efficientnet_lite2",
    "timm-tf_efficientnet_lite3",
    "timm-tf_efficientnet_lite4",
    "timm-resnest14d",
    "timm-resnest26d",
    "timm-resnest50d",
    "timm-resnest101e",
    "timm-resnest200e",
    "timm-resnest269e",
    "timm-resnest50d_4s2x40d",
    "timm-resnest50d_1s4x24d",
    "timm-res2net50_26w_4s",
    "timm-res2net101_26w_4s",
    "timm-res2net50_26w_6s",
    "timm-res2net50_26w_8s",
    "timm-res2net50_48w_2s",
    "timm-res2net50_14w_8s",
    "timm-res2next50",
    "timm-regnetx_002",
    "timm-regnetx_004",
    "timm-regnetx_006",
    "timm-regnetx_008",
    "timm-regnetx_016",
    "timm-regnetx_032",
    "timm-regnetx_040",
    "timm-regnetx_064",
    "timm-regnetx_080",
    "timm-regnetx_120",
    "timm-regnetx_160",
    "timm-regnetx_320",
    "timm-regnety_002",
    "timm-regnety_004",
    "timm-regnety_006",
    "timm-regnety_008",
    "timm-regnety_016",
    "timm-regnety_032",
    "timm-regnety_040",
    "timm-regnety_064",
    "timm-regnety_080",
    "timm-regnety_120",
    "timm-regnety_160",
    "timm-regnety_320",
    "timm-skresnet18",
    "timm-skresnet34",
    "timm-skresnext50_32x4d",
    "timm-mobilenetv3_large_075",
    "timm-mobilenetv3_large_100",
    "timm-mobilenetv3_large_minimal_100",
    "timm-mobilenetv3_small_075",
    "timm-mobilenetv3_small_100",
    "timm-mobilenetv3_small_minimal_100",
    "timm-gernet_s",
    "timm-gernet_m",
    "timm-gernet_l",
]


def build_model(conf):
    model_name = conf['model']['name'].lower().strip()

    # Custom U-Net with pretrained encoder
    if model_name == 'ownunet':
        logging.info("Loading custom OwnUNet model.")

        # If pretrained_encoder_path is specified in config
        pretrained_path = conf['model'].get('pretrained_encoder_path', None)

        if pretrained_path and pretrained_path.lower() != 'none':
            logging.info(f"Loading pretrained encoder weights from {pretrained_path}")
            encoder = Encoder(
                in_channels=conf['model'].get('in_channels', 1),
                base_channels=64
            )
            encoder.load_state_dict(torch.load(pretrained_path, map_location='cpu'))

            model = UNetWithPretrainedEncoder(
                pretrained_encoder=encoder,
                out_channels=conf['model'].get('classes', 1)
            )
        else:
            # No pretrained encoder, build from scratch
            logging.info("No pretrained encoder specified. Initializing randomly.")
            encoder = Encoder(
                in_channels=conf['model'].get('in_channels', 1),
                base_channels=64
            )
            model = UNetWithPretrainedEncoder(
                pretrained_encoder=encoder,
                out_channels=conf['model'].get('classes', 1)
            )
        return model

    # For standard SMP models
    if model_name not in supported_models:
        raise ValueError(
            f"Model {conf['model']['name']} is not supported. "
            f"Supported models are: {list(supported_models.keys())}"
        )
    else:
        logging.info(f"Loading model {model_name} with settings {conf['model']}")

    model_class = supported_models[model_name]
    
    model = model_class(
        encoder_name=conf['model']['encoder_name'],
        encoder_weights=conf['model']['encoder_weights'] if 'None' not in conf['model']['encoder_weights'] else None,
        in_channels=conf['model']['in_channels'],
        classes=conf['model']['classes'],
        activation=conf['model'].get('activation', None),
        #dropout=conf['model'].get('dropout', None)
    )
    return model


# My custom U-Net
class OwnUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))
    
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Modified Encoder to return intermediate features ---
class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)   # shape: B x base_channels x H x W
        x2 = self.enc2(x1)  # shape: B x base_channels*2 x H/2 x W/2
        x3 = self.enc3(x2)  # shape: B x base_channels*4 x H/4 x W/4
        x4 = self.enc4(x3)  # shape: B x base_channels*8 x H/8 x W/8
        return x1, x2, x3, x4


# --- Decoder with skip connections ---
class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=64):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x1, x2, x3, x4):
        d4 = self.up4(x4)                 # Upsample x4 from 1/8 → 1/4 scale
        d4 = torch.cat([d4, x3], dim=1)  # Concat skip connection
        d4 = self.dec4(d4)

        d3 = self.up3(d4)                 # Upsample 1/4 → 1/2 scale
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                 # Upsample 1/2 → original scale
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)

        out = self.final(d2)
        return out


# --- Full U-Net with pretrained encoder ---
class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, pretrained_encoder, out_channels=1):
        super().__init__()
        self.encoder = pretrained_encoder
        self.decoder = Decoder(out_channels=out_channels, base_channels=64)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4)
        return torch.sigmoid(out)
