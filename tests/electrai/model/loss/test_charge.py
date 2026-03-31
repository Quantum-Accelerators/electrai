"""Tests for charge density loss functions."""

from __future__ import annotations

import pytest
import torch

from electrai.model.loss.charge import DensityWeightedNormMAE, NormMAE

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def single_tensor_pair():
    """Single-tensor output/target pair (batch=2, channel=1, 8x8x8)."""
    torch.manual_seed(42)
    target = torch.rand(2, 1, 8, 8, 8) + 0.1
    output = target + 0.01 * torch.randn_like(target)
    return output, target


@pytest.fixture
def list_tensor_pair():
    """List-of-tensors output/target pair (three samples, channel=1, 8x8x8)."""
    torch.manual_seed(42)
    targets = [torch.rand(1, 8, 8, 8) + 0.1 for _ in range(3)]
    outputs = [t + 0.1 * torch.randn_like(t) for t in targets]
    return outputs, targets


# =============================================================================
# DensityWeightedNormMAE Tests
# =============================================================================


class TestDensityWeightedNormMAE:
    def test_produces_scalar(self, single_tensor_pair):
        output, target = single_tensor_pair
        loss = DensityWeightedNormMAE(alpha=1.0, power=1.0)(output, target)
        assert loss.ndim == 0

    def test_alpha_zero_matches_normmae(self, single_tensor_pair):
        output, target = single_tensor_pair
        expected = NormMAE()(output, target)
        actual = DensityWeightedNormMAE(alpha=0.0, power=1.0)(output, target)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_higher_alpha_increases_loss(self, single_tensor_pair):
        output, target = single_tensor_pair
        loss_low = DensityWeightedNormMAE(alpha=0.5)(output, target)
        loss_high = DensityWeightedNormMAE(alpha=5.0)(output, target)
        assert loss_high > loss_low

    def test_list_input_format(self, list_tensor_pair):
        outputs, targets = list_tensor_pair
        loss = DensityWeightedNormMAE(alpha=1.0)(outputs, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_list_alpha_zero_matches_normmae(self, list_tensor_pair):
        outputs, targets = list_tensor_pair
        expected = NormMAE()(outputs, targets)
        actual = DensityWeightedNormMAE(alpha=0.0)(outputs, targets)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_perfect_prediction_gives_zero(self):
        target = torch.rand(1, 1, 4, 4, 4) + 0.01
        loss = DensityWeightedNormMAE(alpha=2.0, power=2.0)(target.clone(), target)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0.0)

    def test_zero_target_no_nan(self):
        target = torch.zeros(1, 1, 4, 4, 4)
        target[0, 0, 2, 2, 2] = 1.0
        output = target + 0.01
        loss = DensityWeightedNormMAE(alpha=1.0)(output, target)
        assert torch.isfinite(loss)

    def test_mismatched_list_lengths_raises(self):
        outputs = [torch.rand(1, 4, 4, 4) for _ in range(3)]
        targets = [torch.rand(1, 4, 4, 4) + 0.1 for _ in range(2)]
        with pytest.raises(ValueError, match="zip"):
            DensityWeightedNormMAE(alpha=1.0)(outputs, targets)

    def test_list_with_non_unity_power(self, list_tensor_pair):
        outputs, targets = list_tensor_pair
        loss = DensityWeightedNormMAE(alpha=1.0, power=2.0)(outputs, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
