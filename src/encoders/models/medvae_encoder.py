import torch
import torch.nn as nn


MEDVAE_MODEL_NAMES = {
    "medvae_4_1_2d",
    "medvae_4_3_2d",
    "medvae_4_4_2d",
    "medvae_8_1_2d",
    "medvae_8_4_2d",
}

# Spatial compression factor per model
COMPRESSION_FACTOR = {
    "medvae_4_1_2d": 4,
    "medvae_4_3_2d": 4,
    "medvae_4_4_2d": 4,
    "medvae_8_1_2d": 8,
    "medvae_8_4_2d": 8,
}

# Channel depth before the final conv_out (block_in = ch * ch_mult[-1])
ENCODER_FEAT_CHANNELS = {
    "medvae_4_1_2d": 512,
    "medvae_4_3_2d": 512,
    "medvae_4_4_2d": 512,
    "medvae_8_1_2d": 512,
    "medvae_8_4_2d": 512,
}

# Input channels each model expects
MODEL_IN_CHANNELS = {
    "medvae_4_1_2d": 1,
    "medvae_4_3_2d": 3,
    "medvae_4_4_2d": 1,
    "medvae_8_1_2d": 1,
    "medvae_8_4_2d": 3,
}


class _MedVAEWrapper(nn.Module):
    def __init__(self, medvae_model, model_name: str, target_dim: int = 512):
        super().__init__()
        self.medvae = medvae_model
        self.model_name = model_name
        self.encoder = medvae_model.encoder
        self.embed_dim = target_dim
        self.compression = COMPRESSION_FACTOR[model_name]
        self.feat_channels = ENCODER_FEAT_CHANNELS[model_name]
        self.grid_size = None

        # Hook to capture mid-block output (before final conv_out)
        self._mid_features = None

        def hook(module, input, output):
            self._mid_features = output.detach()

        self.encoder.mid.block_2.register_forward_hook(hook)

        # Projection from feat_channels to target_dim
        if self.feat_channels != target_dim:
            self.proj = nn.Conv2d(self.feat_channels, target_dim, 1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        self._mid_features = None
        _ = self.encoder(x)
        h = self._mid_features
        h = self.proj(h)
        B, D, H, W = h.shape
        self.grid_size = H
        return h.permute(0, 2, 3, 1).reshape(B, H * W, D)

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False, norm=True):
        B = x.shape[0]
        self._mid_features = None
        _ = self.encoder(x)
        h = self._mid_features
        B, D, H, W = h.shape
        tokens = h.permute(0, 2, 3, 1).reshape(B, H * W, D)
        if return_class_token:
            return [tokens], None
        return [tokens]

    def forward_features(self, x):
        out = self.forward(x)
        return {
            'x_norm_clstoken': None,
            'x_norm_patchtokens': out,
        }

    def to(self, device):
        self.medvae = self.medvae.to(device)
        self.encoder = self.medvae.encoder
        if hasattr(self.proj, 'to'):
            self.proj = self.proj.to(device)
        return self

    def eval(self):
        self.medvae.eval()
        self.encoder.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        self.medvae.requires_grad_(requires_grad)
        return self


def load_medvae(model_name: str, target_dim: int = 512, modality: str = "xray"):
    from medvae import MVAE
    model = MVAE(model_name=model_name, modality=modality)
    model.requires_grad_(False)
    model.eval()
    return _MedVAEWrapper(model.model, model_name, target_dim=target_dim)
