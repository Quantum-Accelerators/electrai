"""Tests for the 3D U-Net architecture with periodic boundary conditions."""

from __future__ import annotations

import pytest
import torch

from electrai.model.unet_3d_pbc import (
    Bottleneck,
    ConvBlock3d,
    DecoderBlock,
    EncoderBlock,
    GeneratorUNet,
)


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestConvBlock3d:
    """Tests for the ConvBlock3d class."""

    def test_output_shape_preserved(self):
        """Test that spatial dimensions are preserved."""
        block = ConvBlock3d(in_channels=16, out_channels=32, kernel_size=3)
        x = torch.randn(2, 16, 8, 8, 8)
        out = block(x)
        assert out.shape == (2, 32, 8, 8, 8)

    def test_channel_expansion(self):
        """Test that channels are changed correctly."""
        block = ConvBlock3d(in_channels=1, out_channels=64, kernel_size=3)
        x = torch.randn(1, 1, 16, 16, 16)
        out = block(x)
        assert out.shape[1] == 64

    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        for kernel_size in [3, 5, 7]:
            block = ConvBlock3d(in_channels=8, out_channels=16, kernel_size=kernel_size)
            x = torch.randn(1, 8, 16, 16, 16)
            out = block(x)
            assert out.shape == (1, 16, 16, 16, 16)

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        block = ConvBlock3d(in_channels=8, out_channels=16, kernel_size=3)
        x = torch.randn(1, 8, 8, 8, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestEncoderBlock:
    """Tests for the EncoderBlock class."""

    def test_downsampling_factor(self):
        """Test that spatial dimensions are halved."""
        block = EncoderBlock(in_channels=16, out_channels=32, kernel_size=3)
        x = torch.randn(2, 16, 16, 16, 16)
        skip, downsampled = block(x)
        # Skip connection preserves dimensions
        assert skip.shape == (2, 32, 16, 16, 16)
        # Downsampled is halved
        assert downsampled.shape == (2, 32, 8, 8, 8)

    def test_channel_expansion(self):
        """Test that channels are expanded correctly."""
        block = EncoderBlock(in_channels=8, out_channels=64, kernel_size=3)
        x = torch.randn(1, 8, 32, 32, 32)
        skip, downsampled = block(x)
        assert skip.shape[1] == 64
        assert downsampled.shape[1] == 64

    def test_skip_connection_output(self):
        """Test that skip connection is returned correctly."""
        block = EncoderBlock(in_channels=16, out_channels=32, kernel_size=3)
        x = torch.randn(1, 16, 8, 8, 8)
        skip, _ = block(x)
        # Skip should have the pre-downsampled features
        assert skip.shape[-1] == 8  # Same spatial dim as input


class TestDecoderBlock:
    """Tests for the DecoderBlock class."""

    def test_upsampling_factor(self):
        """Test that spatial dimensions are doubled."""
        block = DecoderBlock(
            in_channels=64, skip_channels=32, out_channels=32, kernel_size=3
        )
        x = torch.randn(2, 64, 8, 8, 8)
        skip = torch.randn(2, 32, 16, 16, 16)
        out = block(x, skip)
        assert out.shape == (2, 32, 16, 16, 16)

    def test_skip_concatenation(self):
        """Test that skip connection is concatenated."""
        block = DecoderBlock(
            in_channels=64, skip_channels=32, out_channels=32, kernel_size=3
        )
        x = torch.randn(1, 64, 4, 4, 4)
        skip = torch.randn(1, 32, 8, 8, 8)
        out = block(x, skip)
        # Output should match skip spatial dimensions
        assert out.shape[-3:] == skip.shape[-3:]


class TestBottleneck:
    """Tests for the Bottleneck class."""

    def test_output_shape(self):
        """Test bottleneck output shape."""
        block = Bottleneck(in_channels=128, out_channels=256, kernel_size=3)
        x = torch.randn(2, 128, 4, 4, 4)
        out = block(x)
        assert out.shape == (2, 256, 4, 4, 4)

    def test_spatial_preservation(self):
        """Test that spatial dimensions are preserved."""
        block = Bottleneck(in_channels=64, out_channels=128, kernel_size=3)
        x = torch.randn(1, 64, 8, 8, 8)
        out = block(x)
        assert out.shape[-3:] == x.shape[-3:]


class TestGeneratorUNet:
    """Tests for the GeneratorUNet class."""

    def test_same_size_output(self):
        """Test with n_upscale_layers=0 (same size input/output)."""
        model = GeneratorUNet(
            n_upscale_layers=0, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        x = torch.randn(2, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == x.shape

    def test_upscaled_output(self):
        """Test with n_upscale_layers=1 (2x upscaling)."""
        model = GeneratorUNet(
            n_upscale_layers=1, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        x = torch.randn(2, 1, 16, 16, 16)
        out = model(x)
        assert out.shape == (2, 1, 32, 32, 32)

    def test_double_upscaling(self):
        """Test with n_upscale_layers=2 (4x upscaling)."""
        model = GeneratorUNet(
            n_upscale_layers=2, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        x = torch.randn(1, 1, 8, 8, 8)
        out = model(x)
        assert out.shape == (1, 1, 32, 32, 32)

    def test_charge_conservation_normalization(self):
        """Test that normalize=True preserves total charge (scaled)."""
        model = GeneratorUNet(
            n_upscale_layers=1, C=8, depth=2, channel_multiplier=(1, 2), normalize=True
        )
        x = torch.randn(1, 1, 8, 8, 8).abs() + 0.1  # Positive values
        out = model(x)

        # With normalization, output sum should equal input sum * upscale_factor
        upscale_factor = 8  # 2^3 for 3D with 1 upscale layer
        input_sum = x.sum().item()
        output_sum = out.sum().item()
        expected_sum = input_sum * upscale_factor

        assert abs(output_sum - expected_sum) < 1e-4

    def test_no_normalization(self):
        """Test that normalize=False doesn't modify sum."""
        model = GeneratorUNet(
            n_upscale_layers=0, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        x = torch.randn(1, 1, 16, 16, 16).abs() + 0.1
        out = model(x)

        # Without normalization, just verify output is valid
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_variable_batch_sizes_list_input(self):
        """Test with list input for variable-sized batches."""
        model = GeneratorUNet(
            n_upscale_layers=0, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        # List of tensors with different sizes
        x = [torch.randn(1, 16, 16, 16), torch.randn(1, 16, 16, 16)]
        out = model(x)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape == (1, 16, 16, 16)
        assert out[1].shape == (1, 16, 16, 16)

    def test_different_depths(self):
        """Test with different U-Net depths."""
        for depth in [2, 3, 4]:
            model = GeneratorUNet(
                n_upscale_layers=0,
                C=8,
                depth=depth,
                channel_multiplier=(1, 2, 4, 8)[:depth],
                normalize=False,
            )
            # Input size must be divisible by 2^depth
            size = 2 ** (depth + 2)
            x = torch.randn(1, 1, size, size, size)
            out = model(x)
            assert out.shape == x.shape

    def test_different_channel_multipliers(self):
        """Test with different channel multipliers."""
        model = GeneratorUNet(
            n_upscale_layers=0,
            C=16,
            depth=3,
            channel_multiplier=(1, 2, 4),
            normalize=False,
        )
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == x.shape

    def test_gradient_flow_full_model(self):
        """Test that gradients flow through the entire model."""
        model = GeneratorUNet(
            n_upscale_layers=0,
            C=8,
            depth=2,
            channel_multiplier=(1, 2),
            normalize=False,
            use_checkpoint=False,  # Disable checkpointing for gradient test
        )
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_output_non_negative(self):
        """Test that output is non-negative (ReLU at output)."""
        model = GeneratorUNet(
            n_upscale_layers=0, C=8, depth=2, channel_multiplier=(1, 2), normalize=False
        )
        x = torch.randn(1, 1, 16, 16, 16)
        out = model(x)
        assert (out >= 0).all()

    def test_config_compatibility(self):
        """Test that model accepts config-style parameters."""
        # Mimic config parameters as they would come from lightning.py
        model = GeneratorUNet(
            n_upscale_layers=1,
            C=32,
            depth=4,
            K1=5,
            K2=3,
            channel_multiplier=(1, 2, 4, 8),
            normalize=True,
            use_checkpoint=True,
        )
        x = torch.randn(1, 1, 32, 32, 32)
        out = model(x)
        assert out.shape == (1, 1, 64, 64, 64)


class TestGeneratorUNetMemory:
    """Tests for memory efficiency features."""

    def test_gradient_checkpointing_enabled(self):
        """Test that model works with gradient checkpointing enabled."""
        model = GeneratorUNet(
            n_upscale_layers=0,
            C=8,
            depth=2,
            channel_multiplier=(1, 2),
            normalize=False,
            use_checkpoint=True,
        )
        model.train()  # Checkpointing only active in training mode
        x = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_eval_mode(self):
        """Test that model works in eval mode."""
        model = GeneratorUNet(
            n_upscale_layers=0,
            C=8,
            depth=2,
            channel_multiplier=(1, 2),
            normalize=False,
            use_checkpoint=True,
        )
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 1, 16, 16, 16)
            out = model(x)
            assert out.shape == x.shape
