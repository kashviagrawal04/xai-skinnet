import torch
import torch.nn as nn

from .cnn_branch import CNNBranch
from .fusion import AttentionGatedFusion, AttentionGatedFusion3
from .metadata_branch import MetadataBranch
from .vit_branch import ViTBranch


class XAISkinNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.cnn = CNNBranch(m["cnn_backbone"], pretrained=m["pretrained"])
        self.vit = ViTBranch(m["vit_backbone"], pretrained=m["pretrained"])

        # Optional third branch: clinical metadata (age/sex/localization).
        # Off by default -- the paper's/PDF's described architecture is the
        # plain CNN+ViT dual branch. Set `model.use_metadata: true` in the
        # config to enable this as an ablation/extension (see README).
        self.use_metadata = bool(m.get("use_metadata", False))
        if self.use_metadata:
            # Imported here (not at module top) to avoid src.models depending
            # on src.datasets at import time for anything but this constant.
            from src.datasets.ham10000 import METADATA_DIM, encode_metadata_dict

            meta_embed_dim = m.get("metadata_embed_dim", 64)
            self.meta = MetadataBranch(METADATA_DIM, embed_dim=meta_embed_dim)
            self.fusion = AttentionGatedFusion3(
                self.cnn.out_dim, self.vit.out_dim, self.meta.out_dim, m["fusion_dim"]
            )
            # Neutral fallback metadata (age unknown, sex unknown, site
            # unknown) so callers that only pass an image tensor -- e.g.
            # pytorch-grad-cam, which invokes the model as model(x) with no
            # way to thread extra kwargs through its hooks -- still get a
            # well-defined forward pass instead of a crash.
            default_vec = encode_metadata_dict(age=None, sex=None, localization=None)
            self.register_buffer(
                "default_metadata", torch.tensor(default_vec, dtype=torch.float32).unsqueeze(0)
            )
        else:
            self.meta = None
            self.fusion = AttentionGatedFusion(self.cnn.out_dim, self.vit.out_dim, m["fusion_dim"])

        self.classifier = nn.Sequential(
            nn.Linear(m["fusion_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(m["dropout"]),
            nn.Linear(256, cfg["data"]["num_classes"]),
        )

    def _fuse(self, f_cnn, f_vit, metadata):
        if not self.use_metadata:
            return self.fusion(f_cnn, f_vit)
        if metadata is None:
            metadata = self.default_metadata.expand(f_cnn.size(0), -1)
        f_meta = self.meta(metadata)
        return self.fusion(f_cnn, f_vit, f_meta)

    def forward(self, x, metadata=None, return_attn: bool = False):
        f_cnn, _feat_map = self.cnn(x)
        f_vit = self.vit(x)
        fused, alpha = self._fuse(f_cnn, f_vit, metadata)
        logits = self.classifier(fused)
        if return_attn:
            return logits, alpha
        return logits

    def fused_features(self, x, metadata=None):
        """Used by the SHAP pipeline -- returns the fused vector directly."""
        f_cnn, _ = self.cnn(x)
        f_vit = self.vit(x)
        fused, _ = self._fuse(f_cnn, f_vit, metadata)
        return fused

    def freeze_for_warmup(self):
        """
        Warm-up phase (first `train.freeze_epochs` epochs): freeze both
        backbones except their last block, so only the fusion head and
        each branch's final block adapt before full fine-tuning opens up.
        The metadata branch (when enabled) has no pretrained weights to
        protect, so it's left trainable throughout -- it needs the extra
        epochs just as much as the fusion head does.
        """
        for p in self.cnn.backbone.parameters():
            p.requires_grad = False
        for p in self.vit.backbone.parameters():
            p.requires_grad = False
        for p in self.cnn.last_block_params():
            p.requires_grad = True
        for p in self.vit.last_block_params():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
