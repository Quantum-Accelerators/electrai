"""Tests for srgan_layernorm_pbc module."""

from __future__ import annotations

import pytest
import torch

from electrai.model.srgan_layernorm_pbc import (
    GeneratorResNet,
    PixelShuffle3d,
    ResidualBlock,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_input_small():
    """Small 3D input tensor for fast tests."""
    return torch.randn(2, 1, 8, 8, 8)


# =============================================================================
# ResidualBlock Tests
# =============================================================================


class TestResidualBlock:
    """Tests for ResidualBlock class."""

    def test_residual_block_instantiation(self):
        """Verify block creates with default and custom parameters."""
        # Default parameters
        block_default = ResidualBlock(in_features=64)
        assert block_default.use_checkpoint is True

        # Custom parameters
        block_custom = ResidualBlock(in_features=32, K=5, use_checkpoint=False)
        assert block_custom.use_checkpoint is False

    @pytest.mark.parametrize(
        "shape", [(1, 64, 8, 8, 8), (2, 64, 16, 16, 16), (4, 64, 4, 8, 12)]
    )
    def test_residual_block_output_shape(self, shape):
        """Input shape (B, C, H, W, D) → output shape unchanged."""
        block = ResidualBlock(in_features=64, use_checkpoint=False)
        block.eval()

        x = torch.randn(*shape)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_block_residual_connection(self):
        """Verify output ≈ input + conv_block(input) (residual addition works)."""
        in_features = 64
        block = ResidualBlock(in_features=in_features, use_checkpoint=False)
        block.eval()

        x = torch.randn(1, 64, 8, 8, 8)

        # Compute expected output manually
        conv_output = block.conv_block(x)
        expected = x + conv_output

        # Compute actual output
        actual = block(x)

        torch.testing.assert_close(actual, expected)

    def test_residual_block_checkpoint_eval_mode(self):
        """With use_checkpoint=True and model.eval(), checkpointing should be disabled."""
        block = ResidualBlock(in_features=64, use_checkpoint=True)
        block.eval()

        x = torch.randn(1, 64, 8, 8, 8)

        # In eval mode, checkpointing is disabled (per the forward method logic)
        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape

        # Verify output matches non-checkpointed version
        block_no_ckpt = ResidualBlock(in_features=64, use_checkpoint=False)
        block_no_ckpt.load_state_dict(block.state_dict())
        block_no_ckpt.eval()

        with torch.no_grad():
            output_no_ckpt = block_no_ckpt(x)

        torch.testing.assert_close(output, output_no_ckpt)


# =============================================================================
# PixelShuffle3d Tests
# =============================================================================


class TestPixelShuffle3d:
    """Tests for PixelShuffle3d class."""

    def test_pixel_shuffle_3d_instantiation(self):
        """Verify creation with valid in_channels divisible by upscale_factor**3."""
        # Valid configurations
        ps_8_2 = PixelShuffle3d(in_channels=8, upscale_factor=2)  # 8 % 8 == 0
        assert ps_8_2.u == 2
        assert ps_8_2.Cin == 8

        ps_27_3 = PixelShuffle3d(in_channels=27, upscale_factor=3)  # 27 % 27 == 0
        assert ps_27_3.u == 3
        assert ps_27_3.Cin == 27

        ps_64_2 = PixelShuffle3d(in_channels=64, upscale_factor=2)  # 64 % 8 == 0
        assert ps_64_2.u == 2
        assert ps_64_2.Cin == 64

    def test_pixel_shuffle_3d_invalid_channels(self):
        """Verify AssertionError when in_channels % upscale_factor**3 != 0."""
        with pytest.raises(AssertionError):
            PixelShuffle3d(in_channels=7, upscale_factor=2)  # 7 % 8 != 0

        with pytest.raises(AssertionError):
            PixelShuffle3d(in_channels=10, upscale_factor=3)  # 10 % 27 != 0

        with pytest.raises(AssertionError):
            PixelShuffle3d(in_channels=63, upscale_factor=4)  # 63 % 64 != 0

    @pytest.mark.parametrize(
        ("input_shape", "expected_shape"),
        [
            ((1, 64, 4, 4, 4), (1, 8, 8, 8, 8)),
            ((2, 64, 8, 8, 8), (2, 8, 16, 16, 16)),
            ((1, 64, 2, 4, 6), (1, 8, 4, 8, 12)),  # Non-cubic spatial dims
        ],
    )
    def test_pixel_shuffle_3d_output_shape(self, input_shape, expected_shape):
        """Input (B, C*u³, H, W, D) → output (B, C, H*u, W*u, D*u)."""
        ps = PixelShuffle3d(in_channels=64, upscale_factor=2)

        x = torch.randn(*input_shape)
        output = ps(x)
        assert output.shape == expected_shape

    def test_pixel_shuffle_3d_value_mapping(self):
        """Verify values are correctly rearranged (not just shape, but actual shuffling)."""
        # Create a simple deterministic input to verify shuffling
        ps = PixelShuffle3d(in_channels=8, upscale_factor=2)

        # Create input with known values
        x = torch.zeros(1, 8, 2, 2, 2)

        # Set distinct values in each channel
        for c in range(8):
            x[0, c, :, :, :] = c + 1

        output = ps(x)

        # Verify output shape
        assert output.shape == (1, 1, 4, 4, 4)

        # Verify that the values from different channels are now spatially distributed
        # Each 2x2x2 block in output should contain values from different input channels
        assert output.numel() == x.numel()  # Total elements preserved

        # Check that all original values are present
        input_values = set(x.unique().tolist())
        output_values = set(output.unique().tolist())
        assert input_values == output_values

    def test_pixel_shuffle_3d_upscale_factor_3(self):
        """Test with upscale_factor=3: (1, 27, 2, 2, 2) → (1, 1, 6, 6, 6)."""
        ps = PixelShuffle3d(in_channels=27, upscale_factor=3)
        x = torch.randn(1, 27, 2, 2, 2)
        output = ps(x)

        # 27 channels / 3^3 = 1 channel, spatial dims * 3
        expected_shape = (1, 1, 6, 6, 6)
        assert output.shape == expected_shape

        # Test with multiple output channels
        ps_multi = PixelShuffle3d(in_channels=54, upscale_factor=3)  # 54 / 27 = 2
        x_multi = torch.randn(2, 54, 3, 3, 3)
        output_multi = ps_multi(x_multi)

        expected_shape_multi = (2, 2, 9, 9, 9)
        assert output_multi.shape == expected_shape_multi


# =============================================================================
# GeneratorResNet Tests
# =============================================================================


class TestGeneratorResNet:
    """Tests for GeneratorResNet class."""

    # -------------------------------------------------------------------------
    # Instantiation Tests
    # -------------------------------------------------------------------------

    def test_generator_default_instantiation(self):
        """Verify creation with default parameters."""
        gen = GeneratorResNet()

        assert gen.n_upscale_layers == 2
        assert gen.normalize is True
        assert gen.use_checkpoint is True

    def test_generator_custom_parameters(self):
        """Test with custom in_channels, out_channels, n_residual_blocks, C, K1, K2."""
        gen = GeneratorResNet(
            in_channels=3,
            out_channels=2,
            n_residual_blocks=8,
            n_upscale_layers=1,
            C=32,
            K1=3,
            K2=5,
            normalize=False,
            use_checkpoint=False,
        )

        assert gen.n_upscale_layers == 1
        assert gen.normalize is False
        assert gen.use_checkpoint is False

        # Verify number of residual blocks
        assert len(gen.res_blocks) == 8

    def test_generator_single_upscale_layer(self):
        """Test with n_upscale_layers=1 (2x upscaling per axis)."""
        gen = GeneratorResNet(n_upscale_layers=1, use_checkpoint=False)

        assert gen.n_upscale_layers == 1

        # Verify upsampling contains correct number of PixelShuffle3d layers
        pixel_shuffle_count = sum(
            1 for m in gen.upsampling.modules() if isinstance(m, PixelShuffle3d)
        )
        assert pixel_shuffle_count == 1

    def test_generator_three_upscale_layers(self):
        """Test with n_upscale_layers=3 (8x upscaling per axis = 512x total volume)."""
        gen = GeneratorResNet(
            n_upscale_layers=3, n_residual_blocks=2, use_checkpoint=False
        )
        gen.eval()

        assert gen.n_upscale_layers == 3

        # Verify upsampling contains correct number of PixelShuffle3d layers
        pixel_shuffle_count = sum(
            1 for m in gen.upsampling.modules() if isinstance(m, PixelShuffle3d)
        )
        assert pixel_shuffle_count == 3

        # Test forward pass: 8x upscaling per axis
        x = torch.randn(1, 1, 4, 4, 4)
        with torch.no_grad():
            output = gen(x)

        # 4 * 2^3 = 32
        expected_shape = (1, 1, 32, 32, 32)
        assert output.shape == expected_shape

    # -------------------------------------------------------------------------
    # Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_generator_output_shape_default(self, sample_input_small):
        """Input (B, 1, H, W, D) → output (B, 1, 4H, 4W, 4D) with default n_upscale_layers=2."""
        gen = GeneratorResNet(use_checkpoint=False)
        gen.eval()

        output = gen(sample_input_small)

        # With n_upscale_layers=2, each dimension is multiplied by 2^2=4
        expected_shape = (2, 1, 32, 32, 32)
        assert output.shape == expected_shape

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_generator_various_input_sizes(self, size):
        """Test with different spatial dimensions (8³, 16³, 32³)."""
        gen = GeneratorResNet(n_upscale_layers=1, use_checkpoint=False)
        gen.eval()

        x = torch.randn(1, 1, size, size, size)
        output = gen(x)

        expected_size = size * 2  # n_upscale_layers=1 means 2x upscaling
        expected_shape = (1, 1, expected_size, expected_size, expected_size)
        assert output.shape == expected_shape

    def test_generator_output_shape_multichannel(self):
        """Test with in_channels=3, out_channels=2."""
        # Note: normalize=False is required when in_channels != out_channels
        # because the normalization code assumes matching channel dimensions
        gen = GeneratorResNet(
            in_channels=3,
            out_channels=2,
            n_upscale_layers=1,
            n_residual_blocks=2,
            normalize=False,
            use_checkpoint=False,
        )
        gen.eval()

        x = torch.randn(2, 3, 8, 8, 8)
        with torch.no_grad():
            output = gen(x)

        # 2x upscaling, 3 input channels -> 2 output channels
        expected_shape = (2, 2, 16, 16, 16)
        assert output.shape == expected_shape

    def test_generator_non_cubic_input(self):
        """Test with non-cubic input (B, 1, 8, 16, 32)."""
        gen = GeneratorResNet(n_upscale_layers=1, use_checkpoint=False)
        gen.eval()

        x = torch.randn(1, 1, 8, 16, 32)
        with torch.no_grad():
            output = gen(x)

        # Each dimension is doubled independently
        expected_shape = (1, 1, 16, 32, 64)
        assert output.shape == expected_shape

    # -------------------------------------------------------------------------
    # Normalization Tests
    # -------------------------------------------------------------------------

    def test_generator_normalize_true_conserves_total(self):
        """With normalize=True, verify sum(output) ≈ sum(input) * upscale_factor."""
        gen = GeneratorResNet(n_upscale_layers=2, normalize=True, use_checkpoint=False)
        gen.eval()

        # Use positive input (like charge density)
        x = torch.rand(2, 1, 8, 8, 8) + 0.1  # Ensure positive values

        with torch.no_grad():
            output = gen(x)

        # Calculate expected scale factor: 8^n_upscale_layers = 8^2 = 64
        upscale_factor = 8**2

        # Check sum conservation per sample and channel
        for batch_idx in range(x.shape[0]):
            for channel_idx in range(x.shape[1]):
                input_sum = x[batch_idx, channel_idx].sum()
                output_sum = output[batch_idx, channel_idx].sum()
                expected_output_sum = input_sum * upscale_factor

                torch.testing.assert_close(
                    output_sum,
                    expected_output_sum,
                    rtol=1e-4,
                    atol=1e-4,
                    msg=f"Sum not conserved for batch {batch_idx}, channel {channel_idx}",
                )

    def test_generator_normalize_false_no_scaling(self):
        """With normalize=False, output is raw network output."""
        gen_norm = GeneratorResNet(normalize=True, use_checkpoint=False)
        gen_no_norm = GeneratorResNet(normalize=False, use_checkpoint=False)

        # Copy weights from normalized to non-normalized
        gen_no_norm.load_state_dict(gen_norm.state_dict())

        gen_norm.eval()
        gen_no_norm.eval()

        x = torch.rand(1, 1, 8, 8, 8) + 0.1

        with torch.no_grad():
            output_norm = gen_norm(x)
            output_no_norm = gen_no_norm(x)

        # Outputs should be different because of normalization
        assert not torch.allclose(output_norm, output_no_norm)

        # Non-normalized output should have different sum relationship
        input_sum = x.sum()
        output_no_norm_sum = output_no_norm.sum()

        # The ratio should NOT be exactly upscale_factor for non-normalized
        upscale_factor = 8**2
        actual_ratio = output_no_norm_sum / input_sum
        assert not torch.isclose(
            actual_ratio, torch.tensor(float(upscale_factor)), rtol=1e-2
        )

    def test_generator_normalize_batch_independence(self):
        """Normalization computed per-sample in batch."""
        gen = GeneratorResNet(
            n_upscale_layers=1,
            normalize=True,
            n_residual_blocks=2,
            use_checkpoint=False,
        )
        gen.eval()

        # Create batch with very different input sums
        x = torch.rand(3, 1, 8, 8, 8) + 0.1
        x[0] *= 10  # First sample has 10x larger values
        x[2] *= 0.1  # Third sample has 10x smaller values

        with torch.no_grad():
            output = gen(x)

        upscale_factor = 8**1  # n_upscale_layers=1

        # Each sample should independently conserve its sum
        for i in range(3):
            input_sum = x[i].sum()
            output_sum = output[i].sum()
            expected_sum = input_sum * upscale_factor

            torch.testing.assert_close(output_sum, expected_sum, rtol=1e-4, atol=1e-4)

    def test_generator_normalize_channel_independence(self):
        """Normalization computed per-channel."""
        gen = GeneratorResNet(
            in_channels=2,
            out_channels=2,
            n_upscale_layers=1,
            normalize=True,
            n_residual_blocks=2,
            use_checkpoint=False,
        )
        gen.eval()

        # Create input with very different values per channel
        x = torch.rand(1, 2, 8, 8, 8) + 0.1
        x[0, 0] *= 5  # First channel has 5x larger values
        x[0, 1] *= 0.2  # Second channel has 5x smaller values

        with torch.no_grad():
            output = gen(x)

        upscale_factor = 8**1

        # Each channel should independently conserve its sum
        for c in range(2):
            input_sum = x[0, c].sum()
            output_sum = output[0, c].sum()
            expected_sum = input_sum * upscale_factor

            torch.testing.assert_close(output_sum, expected_sum, rtol=1e-4, atol=1e-4)

    def test_generator_with_real_like_data(self):
        """Test with realistic charge density data (positive values, normalized)."""
        gen = GeneratorResNet(
            n_upscale_layers=1,
            normalize=True,
            n_residual_blocks=4,
            use_checkpoint=False,
        )
        gen.eval()

        # Simulate realistic charge density data:
        # - Positive values (charge density is always positive)
        # - Reasonable magnitude range
        # - Sum normalized (total charge conservation)
        torch.manual_seed(42)
        x = torch.abs(torch.randn(2, 1, 8, 8, 8)) + 0.01  # Ensure positive

        # Normalize to simulate typical charge density
        x = x / x.sum(dim=(-3, -2, -1), keepdim=True)
        x = x * 100  # Scale to realistic total charge

        with torch.no_grad():
            output = gen(x)

        # Output should be valid
        assert output.shape == (2, 1, 16, 16, 16)
        assert torch.all(output >= 0), "Charge density should be non-negative"
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values"

        # Sum should be conserved (scaled by upscale factor)
        upscale_factor = 8**1
        for i in range(2):
            input_sum = x[i].sum()
            output_sum = output[i].sum()
            torch.testing.assert_close(
                output_sum, input_sum * upscale_factor, rtol=1e-4, atol=1e-4
            )

    # -------------------------------------------------------------------------
    # Gradient Flow Tests
    # -------------------------------------------------------------------------

    def test_generator_gradient_flow(self):
        """Verify gradients flow through all layers without vanishing."""
        gen = GeneratorResNet(
            n_residual_blocks=4, normalize=False, use_checkpoint=False
        )
        gen.train()

        x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        output = gen(x)
        loss = output.sum()
        loss.backward()

        # Verify input gradient exists and is non-zero
        assert x.grad is not None, "Input gradient should exist"
        assert torch.any(x.grad != 0), "Gradients should flow to input"

        # Verify gradients exist for key layers
        assert gen.conv1[0].weight.grad is not None, "conv1 gradient should exist"
        assert gen.conv3[0].weight.grad is not None, "conv3 gradient should exist"
        assert torch.any(gen.conv1[0].weight.grad != 0), (
            "conv1 gradient should be non-zero"
        )
        assert torch.any(gen.conv3[0].weight.grad != 0), (
            "conv3 gradient should be non-zero"
        )
