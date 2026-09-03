import torch.nn as nn


class MetadataBranch(nn.Module):
    """
    Encodes the fixed-size clinical metadata vector (age/sex/localization —
    see `src/datasets/ham10000.py::encode_metadata_row`) into an embedding
    that can be fused alongside the CNN and ViT branches.

    Deliberately small: metadata carries far less information than the
    image itself, and HAM10000 doesn't have enough samples per
    age/sex/localization combination to justify a deep network here —
    a 2-layer MLP is standard practice for tabular side-inputs in these
    hybrid dermoscopy setups.
    """

    def __init__(self, in_dim: int, embed_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = embed_dim

    def forward(self, x):
        return self.net(x)
