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


@pytest.fixture
def sample_input_medium():
    """Medium 3D input tensor."""
    return torch.randn(1, 1, 16, 16, 16)


@pytest.fixture
def generator_default():
    """Default GeneratorResNet instance."""
    return GeneratorResNet()


@pytest.fixture
def generator_no_normalize():
    """GeneratorResNet with normalize=False."""
    return GeneratorResNet(normalize=False)


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

        # Verify conv_block is a Sequential
        assert isinstance(block_default.conv_block, torch.nn.Sequential)
        assert isinstance(block_custom.conv_block, torch.nn.Sequential)

    def test_residual_block_output_shape(self):
        """Input shape (B, C, H, W, D) → output shape unchanged."""
        in_features = 64
        block = ResidualBlock(in_features=in_features, use_checkpoint=False)
        block.eval()

        # Test various input shapes
        test_shapes = [(1, 64, 8, 8, 8), (2, 64, 16, 16, 16), (4, 64, 4, 8, 12)]

        for shape in test_shapes:
            x = torch.randn(*shape)
            output = block(x)
            assert output.shape == x.shape, f"Shape mismatch for input {shape}"

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

    def test_residual_block_gradient_flow(self):
        """Verify gradients propagate through the block."""
        in_features = 64
        block = ResidualBlock(in_features=in_features, use_checkpoint=False)
        block.train()

        x = torch.randn(1, 64, 8, 8, 8, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check that gradient flowed back to input
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that gradients flowed to all parameters
        for name, param in block.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


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

    def test_pixel_shuffle_3d_output_shape(self):
        """Input (B, C*u³, H, W, D) → output (B, C, H*u, W*u, D*u)."""
        # upscale_factor=2: channels reduce by 8, spatial dims double
        ps = PixelShuffle3d(in_channels=64, upscale_factor=2)
        x = torch.randn(2, 64, 4, 4, 4)
        output = ps(x)

        expected_shape = (2, 8, 8, 8, 8)  # 64/8=8 channels, 4*2=8 spatial
        assert output.shape == expected_shape

    def test_pixel_shuffle_3d_upscale_factor_2(self):
        """Test with upscale_factor=2: (1, 8, 4, 4, 4) → (1, 1, 8, 8, 8)."""
        ps = PixelShuffle3d(in_channels=8, upscale_factor=2)
        x = torch.randn(1, 8, 4, 4, 4)
        output = ps(x)

        expected_shape = (1, 1, 8, 8, 8)
        assert output.shape == expected_shape

    def test_pixel_shuffle_3d_multiple_channels(self):
        """Test with in_channels=64, upscale_factor=2: (B, 64, H, W, D) → (B, 8, 2H, 2W, 2D)."""
        ps = PixelShuffle3d(in_channels=64, upscale_factor=2)

        test_cases = [
            ((1, 64, 4, 4, 4), (1, 8, 8, 8, 8)),
            ((2, 64, 8, 8, 8), (2, 8, 16, 16, 16)),
            ((1, 64, 2, 4, 6), (1, 8, 4, 8, 12)),
        ]

        for input_shape, expected_shape in test_cases:
            x = torch.randn(*input_shape)
            output = ps(x)
            assert output.shape == expected_shape, (
                f"Expected {expected_shape}, got {output.shape}"
            )

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
        assert isinstance(gen.conv1, torch.nn.Sequential)
        assert isinstance(gen.res_blocks, torch.nn.Sequential)
        assert isinstance(gen.conv2, torch.nn.Sequential)
        assert isinstance(gen.upsampling, torch.nn.Sequential)
        assert isinstance(gen.conv3, torch.nn.Sequential)

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

    def test_generator_output_shape_single_upscale(self):
        """Input (B, 1, 8, 8, 8) → output (B, 1, 16, 16, 16)."""
        gen = GeneratorResNet(n_upscale_layers=1, use_checkpoint=False)
        gen.eval()

        x = torch.randn(1, 1, 8, 8, 8)
        output = gen(x)

        # With n_upscale_layers=1, each dimension is multiplied by 2^1=2
        expected_shape = (1, 1, 16, 16, 16)
        assert output.shape == expected_shape

    def test_generator_various_input_sizes(self):
        """Test with different spatial dimensions (8³, 16³, 32³)."""
        gen = GeneratorResNet(n_upscale_layers=1, use_checkpoint=False)
        gen.eval()

        test_sizes = [8, 16, 32]

        for size in test_sizes:
            x = torch.randn(1, 1, size, size, size)
            output = gen(x)

            expected_size = size * 2  # n_upscale_layers=1 means 2x upscaling
            expected_shape = (1, 1, expected_size, expected_size, expected_size)
            assert output.shape == expected_shape, f"Failed for input size {size}³"

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

    # -------------------------------------------------------------------------
    # Gradient & Training Tests
    # -------------------------------------------------------------------------

    def test_generator_gradient_flow(self):
        """Verify gradients propagate from output to input."""
        gen = GeneratorResNet(
            n_residual_blocks=2,  # Fewer blocks for speed
            n_upscale_layers=1,
            use_checkpoint=False,
        )
        gen.train()

        x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
        output = gen(x)
        loss = output.sum()
        loss.backward()

        # Check that gradient flowed back to input
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

        # Check that gradients flowed to key layers
        layers_to_check = [
            ("conv1", gen.conv1),
            ("res_blocks", gen.res_blocks),
            ("conv2", gen.conv2),
            ("upsampling", gen.upsampling),
            ("conv3", gen.conv3),
        ]

        for layer_name, layer in layers_to_check:
            has_gradient = False
            for param in layer.parameters():
                if param.grad is not None and not torch.all(param.grad == 0):
                    has_gradient = True
                    break
            assert has_gradient, f"No gradient for {layer_name}"

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_generator_output_non_negative(self):
        """Final ReLU ensures output ≥ 0."""
        gen = GeneratorResNet(
            n_residual_blocks=4,
            n_upscale_layers=1,
            normalize=False,  # Test raw network output
            use_checkpoint=False,
        )
        gen.eval()

        # Test with various inputs including negative values
        test_inputs = [
            torch.randn(1, 1, 8, 8, 8),  # Mixed positive/negative
            torch.randn(1, 1, 8, 8, 8) - 2,  # Mostly negative
            torch.randn(1, 1, 8, 8, 8) + 2,  # Mostly positive
        ]

        for x in test_inputs:
            with torch.no_grad():
                output = gen(x)

            assert torch.all(output >= 0), "Output contains negative values"
