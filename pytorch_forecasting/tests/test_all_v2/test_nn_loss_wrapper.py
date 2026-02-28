"""
Tests for NNLossWrapper — standard ``nn.Module`` loss adapter for v2 models.
"""

import pytest
import torch
import torch.nn as nn

from pytorch_forecasting.metrics import NNLossWrapper


class TestNNLossWrapperBasic:
    """Unit tests for NNLossWrapper."""

    @pytest.fixture
    def mse_wrapper(self):
        return NNLossWrapper(nn.MSELoss())

    @pytest.fixture
    def l1_wrapper(self):
        return NNLossWrapper(nn.L1Loss())

    @pytest.fixture
    def huber_wrapper(self):
        return NNLossWrapper(nn.HuberLoss(delta=0.5), name="HuberLoss")

    # ---- forward / loss computation ----------------------------------------

    def test_forward_returns_scalar(self, mse_wrapper):
        y_pred = torch.randn(4, 6)  # (batch, horizon)
        y_true = torch.randn(4, 6)
        loss = mse_wrapper(y_pred, y_true)
        assert loss.ndim == 0, "Loss must be a scalar tensor"
        assert loss.requires_grad or y_pred.requires_grad  # differentiable

    def test_forward_with_3d_pred(self, mse_wrapper):
        """3-D predictions (batch, horizon, 1) should be squeezed."""
        y_pred = torch.randn(4, 6, 1, requires_grad=True)
        y_true = torch.randn(4, 6)
        loss = mse_wrapper(y_pred, y_true)
        assert loss.ndim == 0

    def test_forward_with_tuple_target(self, mse_wrapper):
        """A (target, weight) tuple target should silently discard weights."""
        y_pred = torch.randn(4, 6)
        y_true = torch.randn(4, 6)
        weight = torch.ones(4, 6)
        loss = mse_wrapper(y_pred, (y_true, weight))
        assert loss.ndim == 0

    def test_l1_wrapper(self, l1_wrapper):
        y_pred = torch.ones(2, 3)
        y_true = torch.zeros(2, 3)
        loss = l1_wrapper(y_pred, y_true)
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_custom_name(self, huber_wrapper):
        assert huber_wrapper.name == "HuberLoss"

    def test_default_name_from_class(self, mse_wrapper):
        assert mse_wrapper.name == "MSELoss"

    # ---- to_prediction -------------------------------------------------------

    def test_to_prediction_2d(self, mse_wrapper):
        y_pred = torch.randn(4, 6)
        out = mse_wrapper.to_prediction(y_pred)
        assert out.shape == (4, 6)

    def test_to_prediction_3d_singleton(self, mse_wrapper):
        y_pred = torch.randn(4, 6, 1)
        out = mse_wrapper.to_prediction(y_pred)
        assert out.shape == (4, 6)

    def test_to_prediction_3d_non_singleton_unchanged(self, mse_wrapper):
        """3-D with last dim > 1 stays unchanged (caller's responsibility)."""
        y_pred = torch.randn(4, 6, 3)
        out = mse_wrapper.to_prediction(y_pred)
        assert out.shape == (4, 6, 3)

    # ---- to_quantiles --------------------------------------------------------

    def test_to_quantiles_2d(self, mse_wrapper):
        y_pred = torch.randn(4, 6)
        out = mse_wrapper.to_quantiles(y_pred)
        assert out.shape == (4, 6, 1), "Expected (batch, horizon, 1)"

    def test_to_quantiles_3d_singleton(self, mse_wrapper):
        y_pred = torch.randn(4, 6, 1)
        out = mse_wrapper.to_quantiles(y_pred)
        assert out.shape == (4, 6, 1)

    # ---- torchmetrics state --------------------------------------------------

    def test_compute_accumulates_batches(self, mse_wrapper):
        mse_wrapper.reset()
        y1 = torch.zeros(2, 3)
        y2 = torch.ones(2, 3)
        mse_wrapper.update(y1, y1)
        mse_wrapper.update(y2, y2)
        computed = mse_wrapper.compute()
        assert computed == 0.0

    def test_repr(self, mse_wrapper):
        r = repr(mse_wrapper)
        assert "NNLossWrapper" in r
        assert "MSELoss" in r

    # ---- integration: use as BaseModel loss ----------------------------------

    def test_wraps_any_nn_module(self):
        """Any callable nn.Module loss should be accepted."""
        losses = [
            nn.MSELoss(),
            nn.L1Loss(),
            nn.HuberLoss(),
            nn.SmoothL1Loss(),
        ]
        for fn in losses:
            wrapper = NNLossWrapper(fn)
            y_pred = torch.randn(2, 4)
            y_true = torch.randn(2, 4)
            loss_val = wrapper(y_pred, y_true)
            assert loss_val.ndim == 0, f"Scalar expected for {fn}"

    # ---- regression: double-update bug ----------------------------------------

    def test_forward_does_not_double_count(self, mse_wrapper):
        """Calling forward() once must increment _num_batches by exactly 1."""
        mse_wrapper.reset()
        y_pred = torch.randn(4, 6)
        y_true = torch.randn(4, 6)
        _ = mse_wrapper(y_pred, y_true)  # one forward pass
        assert mse_wrapper._num_batches.item() == 1, (
            f"Expected _num_batches=1 after one forward(), "
            f"got {mse_wrapper._num_batches.item()} (double-update bug?)"
        )

    # ---- loss() method ---------------------------------------------------------

    def test_loss_method_delegates_to_wrapped(self, mse_wrapper):
        """loss() should return the same value as the wrapped nn.Module."""
        y_pred = torch.randn(4, 6)
        y_true = torch.randn(4, 6)
        expected = mse_wrapper.loss_fn(y_pred, y_true)
        actual = mse_wrapper.loss(y_pred, y_true)
        assert torch.isclose(expected, actual), (
            f"loss() mismatch: expected {expected.item():.6f}, got {actual.item():.6f}"
        )

    def test_loss_method_does_not_update_accumulator(self, mse_wrapper):
        """loss() must NOT change the running accumulator state."""
        mse_wrapper.reset()
        y_pred = torch.randn(4, 6)
        y_true = torch.randn(4, 6)
        before = mse_wrapper._num_batches.item()
        _ = mse_wrapper.loss(y_pred, y_true)
        assert mse_wrapper._num_batches.item() == before, (
            "loss() should not update the running accumulator"
        )

    # ---- user requested tests --------------------------------------------------

    def test_forward_calls_loss_once(self):
        class CountLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x, y):
                self.calls += 1
                return torch.mean((x - y) ** 2)

        cnt = CountLoss()
        wrapper = NNLossWrapper(cnt)

        y_pred = torch.randn(4, 6)
        y_true = torch.randn(4, 6)

        wrapper(y_pred, y_true)

        assert cnt.calls == 1

    def test_forward_device_alignment(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        wrapper = NNLossWrapper(nn.MSELoss())

        y_pred = torch.randn(4, 6).cuda()
        y_true = torch.randn(4, 6)  # CPU

        loss = wrapper(y_pred, y_true)

        assert loss.device == y_pred.device
