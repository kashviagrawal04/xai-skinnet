import timm
import torch.nn as nn


class ViTBranch(nn.Module):
    """
    ViT-B/16 feature extractor. Returns the [CLS] token (768-D) as the
    global feature vector F_ViT.

    Note: timm's `forward_features` for ViT models returns the full token
    sequence [B, N+1, D] on most timm versions (patch tokens + CLS at
    index 0). If your installed timm version instead returns an already
    class-token variant, this still works since we handle both shapes.
    """

    def __init__(self, backbone_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features  # 768 for ViT-B/16

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        if tokens.dim() == 3:
            cls_token = tokens[:, 0]
        else:
            cls_token = tokens
        return cls_token

    def last_block_params(self):
        """Params to keep trainable during the frozen-backbone warm-up phase."""
        return list(self.backbone.blocks[-1].parameters())
