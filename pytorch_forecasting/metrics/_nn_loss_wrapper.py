"""
Adapter wrapping standard ``torch.nn`` loss functions as v2 ``Metric``-compatible losses.

This allows any ``nn.Module`` loss (e.g. ``nn.MSELoss``, ``nn.L1Loss``,
``nn.HuberLoss``) to be passed directly as the ``loss`` argument to v2
``BaseModel`` subclasses without requiring a custom ``Metric`` subclass.

Example
-------
>>> import torch.nn as nn
>>> from pytorch_forecasting.metrics import NNLossWrapper
>>> loss = NNLossWrapper(nn.MSELoss())
"""

from typing import Optional

import torch
import torch.nn as nn

from pytorch_forecasting.metrics.base_metrics import Metric


class NNLossWrapper(Metric):
    """Wrap a standard ``torch.nn`` loss module as a v2-compatible ``Metric``.

    Enables any :class:`~torch.nn.Module` loss function — such as
    :class:`~torch.nn.MSELoss`, :class:`~torch.nn.L1Loss`, or
    :class:`~torch.nn.HuberLoss` — to be used as a drop-in replacement for
    native :class:`~pytorch_forecasting.metrics.Metric` instances in v2
    :class:`~pytorch_forecasting.models.base.BaseModel` subclasses.

    .. note::
        The wrapper automatically aligns the `device` and `dtype` of the target
        tensors to match the prediction tensors before passing them to the
        underlying ``loss_fn``. This prevents device mismatch runtime errors
        (e.g. GPU predictions vs CPU targets).

    Parameters
    ----------
    loss_fn : nn.Module
        Any ``torch.nn`` loss module whose ``forward(input, target)`` signature
        accepts two tensors and returns a scalar tensor.
    name : str, optional
        Human-readable name.  Defaults to the class name of *loss_fn*.

    Examples
    --------
    Basic usage with MSE:

    .. code-block:: python

        import torch.nn as nn
        from pytorch_forecasting.metrics import NNLossWrapper
        from pytorch_forecasting.metrics import SMAPE

        mse_loss = NNLossWrapper(nn.MSELoss())

        # Use as the loss argument in any v2 model
        # model = TFT(loss=mse_loss, ...)

    Huber loss with delta:

    .. code-block:: python

        huber = NNLossWrapper(nn.HuberLoss(delta=0.5), name="HuberLoss")
    """

    full_state_update: bool = False
    higher_is_better: bool = False
    is_differentiable: bool = True

    def __init__(self, loss_fn: nn.Module, name: Optional[str] = None) -> None:
        if name is None:
            name = type(loss_fn).__name__
        super().__init__(name=name)
        self.loss_fn = loss_fn
        # Accumulator state for torchmetrics compatibility
        self.add_state("_total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("_num_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    # ------------------------------------------------------------------
    # torchmetrics protocol
    # ------------------------------------------------------------------

    def update(self, y_pred: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore[override]
        """Accumulate loss for a single batch.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions.  Shape ``(batch, horizon, output_size)`` or
            ``(batch, horizon)``.
        target : torch.Tensor or tuple
            Ground-truth targets.  A ``(target, weight)`` tuple is accepted;
            weights are ignored (standard ``nn`` losses do not support them).
        """
        if isinstance(target, (tuple, list)):
            target = target[0]  # discard optional weight

        y_pred = self.to_prediction(y_pred)
        # Align shapes: broadcast target to match y_pred if needed
        target = target.to(dtype=y_pred.dtype, device=y_pred.device)

        batch_loss = self.loss_fn(y_pred, target)
        self._total_loss += batch_loss.detach()
        self._num_batches += 1

    def compute(self) -> torch.Tensor:
        """Return mean loss over all accumulated batches."""
        if self._num_batches == 0:
            return self._total_loss.new_tensor(0.0)
        return self._total_loss / self._num_batches

    @torch.jit.unused
    def forward(  # type: ignore[override]
        self, y_pred: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute and return the loss for back-propagation.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions.
        target : torch.Tensor or tuple
            Ground-truth targets.  A ``(target, weight)`` tuple is accepted;
            weights are ignored.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if isinstance(target, (tuple, list)):
            target = target[0]

        y_pred_point = self.to_prediction(y_pred)
        target = target.to(dtype=y_pred_point.dtype, device=y_pred_point.device)

        # Update internal accumulator state once per batch
        loss_val = self.loss_fn(y_pred_point, target)
        self._total_loss += loss_val.detach()
        self._num_batches += 1

        return loss_val

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Delegate to the wrapped ``nn.Module`` loss.

        Satisfies the abstract ``Metric.loss()`` interface.  Unlike
        :meth:`forward`, this method does **not** update the running
        accumulator — use it when you need the raw loss value without
        affecting ``compute()``.

        Parameters
        ----------
        y_pred : torch.Tensor
            Model predictions.
        target : torch.Tensor or tuple
            Ground-truth targets.  A ``(target, weight)`` tuple is accepted;
            weights are ignored.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if isinstance(target, (tuple, list)):
            target = target[0]
        y_pred_point = self.to_prediction(y_pred)
        target = target.to(dtype=y_pred_point.dtype, device=y_pred_point.device)
        return self.loss_fn(y_pred_point, target)

    # ------------------------------------------------------------------
    # Prediction conversion helpers
    # ------------------------------------------------------------------

    def to_prediction(self, y_pred: torch.Tensor, **kwargs) -> torch.Tensor:
        """Convert raw model output to a point prediction.

        If *y_pred* has three dimensions ``(batch, horizon, 1)`` the trailing
        singleton is squeezed.  Tensors already of shape ``(batch, horizon)``
        are returned unchanged.

        Parameters
        ----------
        y_pred : torch.Tensor
            Raw model output.

        Returns
        -------
        torch.Tensor
            Point prediction of shape ``(batch, horizon)``.
        """
        if y_pred.ndim == 3 and y_pred.size(-1) == 1:
            return y_pred.squeeze(-1)
        return y_pred

    def to_quantiles(
        self,
        y_pred: torch.Tensor,
        quantiles: Optional[list] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Wrap point prediction in an extra dimension for API compatibility.

        Standard ``nn.Module`` losses are point-estimate losses and do not
        produce distributional outputs.  This method expands the point
        prediction to shape ``(batch, horizon, 1)`` so it can flow through
        the same code-paths that expect quantile tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            Raw model output.
        quantiles : list of float, optional
            Ignored — present only for interface compatibility.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, horizon, 1)``.
        """
        point = self.to_prediction(y_pred)
        return point.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"NNLossWrapper({repr(self.loss_fn)})"

    def extra_repr(self) -> str:  # type: ignore[override]
        return repr(self.loss_fn)
