import timm
import torch.nn as nn


class CNNBranch(nn.Module):
    """
    EfficientNet-B4 feature extractor.

    Returns both the pooled feature vector (used for fusion) and the raw
    feature map (used by Grad-CAM). `global_pool=''` and `num_classes=0`
    strip timm's own head so we control pooling ourselves.
    """

    def __init__(self, backbone_name: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = self.backbone.num_features  # 1792 for efficientnet_b4

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)  # [B, C, H, W]
        pooled = self.pool(feat_map).flatten(1)  # [B, C]
        return pooled, feat_map

    def grad_cam_target_layer(self):
        """The conv layer to hook for Grad-CAM — the final 1x1 conv before pooling."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        return self.backbone.blocks[-1]

    def last_block_params(self):
        """Params to keep trainable during the frozen-backbone warm-up phase."""
        if hasattr(self.backbone, "conv_head"):
            return list(self.backbone.conv_head.parameters())
        return list(self.backbone.blocks[-1].parameters())
