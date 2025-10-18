import segmentation_models_pytorch as smp
import torch
import logging
import torch.nn as nn
from deep_closing_modules import Unet_MIM, Simple_Point_Erosion_Module

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
    "mit_b2",
    "swin_base_patch4_window7_224",
    "maxvit_base_tf_512",
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
    "swin_s3_small_224",
]


def build_model(conf):
    model_name = conf['model']['name'].lower().strip()
    feature_dirs = conf['dataset']['feature_dirs']
    in_channels = len(feature_dirs)
    if 'mask' in feature_dirs:
        in_channels -= 1
    if 'distance' in feature_dirs:
        in_channels -= 1

    # --- NEW LOGIC FOR DEEP CLOSING MODELS ---
    # Add a special case for the masked autoencoder pre-training step
    if model_name == 'unet_mim':
        logging.info("Building Unet_MIM for pre-training.")
        model = Unet_MIM(
            in_channels=conf['model'].get('in_channels', 1),
            out_channels=conf['model'].get('out_channels', 1),
            mask_ratio=tuple(conf['model']['mask_ratio']),
            patch_size=(conf['model']['patch_size_h'], conf['model']['patch_size_w'])
        )
        return model
        
    # Add a special case for building the final DeepClosing model
    if model_name == 'deepclosing_refiner':
        logging.info("Building DeepClosingRefiner model.")
        # Load the pre-trained autoencoder (Deep Dilation module)
        ae_path = conf['model']['autoencoder_path']
        if not os.path.exists(ae_path):
            raise FileNotFoundError(f"Pre-trained autoencoder not found at: {ae_path}")
        logging.info(f"Loading pre-trained autoencoder from {ae_path}")
        autoencoder = torch.load(ae_path, map_location='cpu')
        
        model = DeepClosingRefiner(autoencoder)
        return model
    # Custom U-Net with pretrained encoder
    if model_name == 'ownunet':
        logging.info("Loading custom OwnUNet model.")
        model = OwnUNet(
            in_channels=in_channels,
            out_channels=conf['model']['classes']
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
        aux_params = dict(dropout=conf['model'].get('dropout', None), classes=conf['model']['classes'])
        model = model_class(
            encoder_name=conf['model']['encoder_name'],
            encoder_weights=conf['model']['encoder_weights'],
            in_channels=in_channels,
            classes=conf['model']['classes'],
            activation=conf['model'].get('activation', None),
            aux_params=aux_params
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

        return self.final(d1)
    
# =========================================================================
#  ------ NEW DEEP CLOSING REFINER MODEL ------
# =========================================================================
# This is the final model for the refinement stage.

class DeepClosingRefiner(nn.Module):
    """
    Encapsulates the full Deep Closing pipeline:
    1. A pre-trained autoencoder acts as a "Deep Dilation" module.
    2. A Simple Point Erosion module acts as a "Deep Erosion" module.
    
    This model is NOT trained. It's used for inference only.
    """
    def __init__(self, pretrained_autoencoder, device=torch.device("cuda:0")):
        super().__init__()
        self.device = device
        
        # 1. The Deep Dilation module is the pre-trained Unet_MIM's internal network.
        #    We freeze it as it should not be trained further.
        self.deep_dilation_net = pretrained_autoencoder.net
        for param in self.deep_dilation_net.parameters():
            param.requires_grad = False
        
        # 2. The Deep Erosion module
        self.erosion_module = Simple_Point_Erosion_Module(device=self.device)
        
        self.to(self.device)
        self.eval() # This module is always in evaluation mode.

    @torch.no_grad()
    def forward(self, initial_prediction):
        """
        Takes an initial, possibly fragmented segmentation mask and refines it.
        
        Args:
            initial_prediction (torch.Tensor): A binary tensor of shape (B, 1, H, W)
                                               from a standard segmentation model.
        Returns:
            dict: A dictionary containing the final closed prediction and intermediate steps.
        """
        initial_prediction = initial_prediction.to(self.device)
        
        # --- Step 1: Deep Dilation ---
        # The autoencoder, when given a fragmented mask, will try to reconstruct
        # the full shape, effectively "filling in the gaps" or dilating it.
        dilated_logits = self.deep_dilation_net(initial_prediction)
        dilated_prob = torch.sigmoid(dilated_logits)
        
        # Binarize the dilated output
        dilated_binary = (dilated_prob > 0.5).float()
        
        # --- Step 2: Identify Regions to Erode ---
        # The mask for erosion (M_T in the paper) are the pixels that were
        # added by the dilation step.
        erosion_mask = (dilated_binary - initial_prediction).clamp(min=0)
        
        # --- Step 3: Deep Erosion ---
        # Apply the topology-preserving erosion only on the newly added regions.
        final_closed_mask = self.erosion_module.erode(
            T=dilated_binary, 
            M_T=erosion_mask, 
            max_k=100 # A high number to ensure it runs to completion
        )
        
        return {
            "initial_prediction": initial_prediction,
            "deep_dilation_output": dilated_binary,
            "erosion_mask": erosion_mask,
            "final_closed_mask": final_closed_mask
        }