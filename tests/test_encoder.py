"""Tests for chemvision.models.encoder module."""

from __future__ import annotations

import pytest
from PIL import Image as PILImage

from chemvision.models.encoder import DynamicResolutionEncoder, EncoderConfig, PatchEmbeddings


# ---------------------------------------------------------------------------
# EncoderConfig schema
# ---------------------------------------------------------------------------


def test_encoder_config_defaults() -> None:
    cfg = EncoderConfig(vision_model_name_or_path="dummy/model")
    assert cfg.grid_size == 4
    assert cfg.high_resolution == 672
    assert cfg.low_resolution == 224
    assert cfg.saliency_top_k == pytest.approx(0.25)
    assert cfg.dtype == "bfloat16"


def test_encoder_config_custom() -> None:
    cfg = EncoderConfig(
        vision_model_name_or_path="openai/clip",
        grid_size=2,
        high_resolution=512,
        low_resolution=128,
        saliency_top_k=0.5,
        device="cpu",
    )
    assert cfg.grid_size == 2
    assert cfg.saliency_top_k == pytest.approx(0.5)
    assert cfg.device == "cpu"


def test_encoder_config_grid_size_must_be_positive() -> None:
    with pytest.raises(Exception):
        EncoderConfig(vision_model_name_or_path="dummy", grid_size=0)


def test_encoder_config_saliency_top_k_bounds() -> None:
    with pytest.raises(Exception):
        EncoderConfig(vision_model_name_or_path="dummy", saliency_top_k=1.5)
    with pytest.raises(Exception):
        EncoderConfig(vision_model_name_or_path="dummy", saliency_top_k=-0.1)


# ---------------------------------------------------------------------------
# PatchEmbeddings dataclass
# ---------------------------------------------------------------------------


def test_patch_embeddings_num_patches() -> None:
    dummy_tensor = object()
    pe = PatchEmbeddings(
        embeddings=dummy_tensor,
        saliency_scores=[0.1, 0.2, 0.3, 0.4],
        resolutions=[224, 224, 672, 672],
        grid_size=2,
    )
    assert pe.num_patches == 4


def test_patch_embeddings_num_patches_3x3() -> None:
    pe = PatchEmbeddings(
        embeddings=None,
        saliency_scores=[float(i) for i in range(9)],
        resolutions=[224] * 9,
        grid_size=3,
    )
    assert pe.num_patches == 9


# ---------------------------------------------------------------------------
# Helpers that work without a loaded model
# ---------------------------------------------------------------------------


def _make_encoder(grid_size: int = 2) -> DynamicResolutionEncoder:
    cfg = EncoderConfig(vision_model_name_or_path="dummy", grid_size=grid_size, device="cpu")
    return DynamicResolutionEncoder(cfg)


# _split_patches


def test_split_patches_count_2x2() -> None:
    enc = _make_encoder(grid_size=2)
    image = PILImage.new("RGB", (100, 100), color=(128, 0, 0))
    patches = enc._split_patches(image)
    assert len(patches) == 4


def test_split_patches_count_3x3() -> None:
    enc = _make_encoder(grid_size=3)
    image = PILImage.new("RGB", (90, 90), color=(0, 128, 0))
    patches = enc._split_patches(image)
    assert len(patches) == 9


def test_split_patches_dimensions() -> None:
    enc = _make_encoder(grid_size=2)
    image = PILImage.new("RGB", (100, 80), color=(0, 0, 128))
    patches = enc._split_patches(image)
    # Each patch should be 50×40 (width × height)
    for p in patches:
        assert p.size == (50, 40)


def test_split_patches_are_pil_images() -> None:
    enc = _make_encoder(grid_size=2)
    image = PILImage.new("RGB", (64, 64))
    patches = enc._split_patches(image)
    for p in patches:
        assert isinstance(p, PILImage.Image)


# _assign_resolutions


def test_assign_resolutions_top_25_percent() -> None:
    cfg = EncoderConfig(
        vision_model_name_or_path="dummy",
        grid_size=4,
        saliency_top_k=0.25,
        high_resolution=672,
        low_resolution=224,
    )
    enc = DynamicResolutionEncoder(cfg)
    # 16 patches, top 25% = 4 at high resolution
    scores = list(range(16))  # 0..15; patch 15 is most salient
    resolutions = enc._assign_resolutions(scores)
    assert resolutions.count(672) == 4
    assert resolutions.count(224) == 12


def test_assign_resolutions_highest_gets_high_res() -> None:
    cfg = EncoderConfig(
        vision_model_name_or_path="dummy",
        grid_size=2,
        saliency_top_k=0.25,  # 25% of 4 = 1 patch at high res
        high_resolution=512,
        low_resolution=128,
    )
    enc = DynamicResolutionEncoder(cfg)
    scores = [0.1, 0.9, 0.3, 0.5]  # patch 1 is most salient
    resolutions = enc._assign_resolutions(scores)
    assert resolutions[1] == 512
    assert resolutions[0] == 128
    assert resolutions[2] == 128
    assert resolutions[3] == 128


def test_assign_resolutions_at_least_one_high() -> None:
    """Even with a very small top_k fraction, at least one patch gets high res."""
    cfg = EncoderConfig(
        vision_model_name_or_path="dummy",
        grid_size=2,
        saliency_top_k=0.01,
        high_resolution=512,
        low_resolution=128,
    )
    enc = DynamicResolutionEncoder(cfg)
    scores = [0.1, 0.2, 0.3, 0.4]
    resolutions = enc._assign_resolutions(scores)
    assert resolutions.count(512) >= 1


def test_assign_resolutions_length_matches_scores() -> None:
    enc = _make_encoder(grid_size=3)
    scores = [float(i) for i in range(9)]
    resolutions = enc._assign_resolutions(scores)
    assert len(resolutions) == 9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_encode_before_load_raises() -> None:
    enc = _make_encoder()
    image = PILImage.new("RGB", (100, 100))
    with pytest.raises(RuntimeError, match="load"):
        enc.encode(image)


def test_repr_shows_unloaded() -> None:
    enc = _make_encoder()
    assert "unloaded" in repr(enc)
    assert "dummy" in repr(enc)
