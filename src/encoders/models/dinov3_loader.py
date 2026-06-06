import os
import warnings
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision import transforms


@contextmanager
def _rank0_first():
    """Gate torch.hub download so only rank 0 fetches, others wait then read from cache."""
    initialized = dist.is_initialized()
    rank = dist.get_rank() if initialized else 0
    if initialized and rank != 0:
        dist.barrier()
    try:
        yield
    finally:
        if initialized and rank == 0:
            dist.barrier()


def make_dinov3_transform(resize_size: int = 224):
    to_tensor = transforms.Lambda(lambda x: x / 255.)
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


DINOV3_HUB_REF = "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[3] / "pretrained_models" / "encoders" / "dinov3"

MODEL_NAMES = {
    'dinov3_vits16',
    "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
}
SHA_CHECKSUM = {
    "dinov3_vits16": "08c60483",
    "dinov3_vits16plus": "4057cbaa",
    "dinov3_vitb16": "73cec8be",
    "dinov3_vitl16": "8aa4cbdd",
    "dinov3_vith16plus": "7c1da9a5",
    "dinov3_vit7b16": "a955f4ea",
}

TIMM_MODEL_MAP = {
    "dinov3_vits16": "vit_small_patch16_dinov3",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3",
    "dinov3_vitb16": "vit_base_patch16_dinov3",
    "dinov3_vitl16": "vit_large_patch16_dinov3",
    "dinov3_vith16plus": "vit_huge_plus_patch16_dinov3",
    "dinov3_vit7b16": "vit_7b_patch16_dinov3",
}


class _TimmDinov3Wrapper(nn.Module):
    """Wraps a timm DINOv3 model to match the torch.hub interface (forward_features returns dict)."""

    def __init__(self, timm_model):
        super().__init__()
        self.timm_model = timm_model
        self.embed_dim = timm_model.embed_dim

    @property
    def norm(self):
        return self.timm_model.norm

    @norm.setter
    def norm(self, value):
        self.timm_model.norm = value

    def forward_features(self, x):
        out = self.timm_model.forward_features(x)
        n_prefix = self.timm_model.num_prefix_tokens
        cls_token = out[:, 0]
        patch_tokens = out[:, n_prefix:]
        return {
            'x_norm_clstoken': cls_token,
            'x_norm_patchtokens': patch_tokens,
        }

    def get_intermediate_layers(self, x, n, reshape=False, return_class_token=False, norm=True):
        # timm Eva uses forward_intermediates (not get_intermediate_layers)
        # Output from forward_intermediates is list of (B, C, H, W) tensors
        _, intermediates = self.timm_model.forward_intermediates(
            x, indices=n, norm=norm, return_prefix_tokens=return_class_token,
        )
        # Convert from NCHW to NL(C) format expected by the caller
        result = []
        for h in intermediates:
            b, c, h_dim, w_dim = h.shape
            h = h.reshape(b, c, -1).transpose(1, 2)  # (B, N, C)
            result.append(h)
        return result

    def forward(self, x):
        return self.timm_model(x)

    def to(self, device):
        self.timm_model = self.timm_model.to(device)
        return self

    def eval(self):
        self.timm_model.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        self.timm_model.requires_grad_(requires_grad)
        return self


def _load_via_timm(model_name):
    """Load DINOv3 via timm (HuggingFace-hosted weights)."""
    import timm
    timm_name = TIMM_MODEL_MAP[model_name]
    model = timm.create_model(timm_name, pretrained=True)
    model.eval()
    return _TimmDinov3Wrapper(model)


def load_dinov3(model_name, force_download: bool = False):
    """Load a DINOv3 model by name.

    Tries torch.hub first (local cache or Meta's dl.fbaipublicfiles.com),
    falls back to timm (HuggingFace-hosted weights).

    Args:
        model_name: e.g. "dinov3_vitb16"
        force_download: If True, skip local cache and download from official URL.
    """
    assert model_name in MODEL_NAMES
    ckpt_dir = os.environ.get("DINOV3_CKPT_DIR", str(DEFAULT_CKPT_DIR))
    weights_path = os.path.join(ckpt_dir, f"{model_name}_pretrain_lvd1689m-{SHA_CHECKSUM[model_name]}.pth")
    weight_kwargs = {}
    if not force_download and os.path.isfile(weights_path):
        weight_kwargs['weights'] = weights_path
    repo_dir = os.environ.get("DINOV3_REPO_DIR")
    with _rank0_first():
        if weight_kwargs or (repo_dir and os.path.isfile(os.path.join(repo_dir, "hubconf.py"))):
            try:
                if repo_dir and os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
                    return torch.hub.load(
                        repo_dir, model_name, source="local",
                        trust_repo=True, skip_validation=True, **weight_kwargs,
                    )
                return torch.hub.load(
                    DINOV3_HUB_REF, model_name, source="github",
                    trust_repo=True, skip_validation=True, **weight_kwargs,
                )
            except Exception as e:
                warnings.warn(f"torch.hub load failed ({e}), falling back to timm")
        return _load_via_timm(model_name)
