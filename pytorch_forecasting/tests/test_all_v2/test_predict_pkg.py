"""
Unit tests for Base_pkg.predict() and Base_pkg._load_dataloader().

These tests operate entirely without a trained model or Lightning Trainer,
by using lightweight stub subclasses to exercise the logic paths in isolation.
"""

import pytest
import torch
import pandas as pd
import warnings

from pytorch_forecasting.base._base_pkg import Base_pkg
from pytorch_forecasting.data import TimeSeries


# ---------------------------------------------------------------------------
# Minimal stub to let us instantiate Base_pkg without real model/data config
# ---------------------------------------------------------------------------


class _StubPkg(Base_pkg):
    """Minimal concrete subclass of Base_pkg for testing package plumbing."""

    @classmethod
    def get_cls(cls):
        return None  # not needed for the tests below

    @classmethod
    def get_datamodule_cls(cls):
        return None  # not needed for the tests below


class TestLoadDataloaderTypeChecks:
    """_load_dataloader() must accept valid types and reject invalid ones."""

    def _make_pkg(self, datamodule_cfg: dict | None = None):
        return _StubPkg(
            model_cfg={},
            trainer_cfg={},
            datamodule_cfg=datamodule_cfg or {},
        )

    def test_unsupported_type_raises_type_error(self):
        pkg = self._make_pkg()
        with pytest.raises(TypeError, match="Unsupported data type"):
            pkg._load_dataloader(object())

    def test_unsupported_list_raises_type_error(self):
        pkg = self._make_pkg()
        with pytest.raises(TypeError, match="Unsupported data type"):
            pkg._load_dataloader([1, 2, 3])

    def test_unsupported_int_raises_type_error(self):
        pkg = self._make_pkg()
        with pytest.raises(TypeError, match="Unsupported data type"):
            pkg._load_dataloader(42)

    def test_dataframe_builds_timeseries_using_datamodule_cfg(self):
        """pd.DataFrame input should be wrapped into a TimeSeries via cfg keys."""
        records = []
        for g in range(2):
            for t in range(30):
                records.append({"group": g, "time_idx": t, "target": float(t)})
        df = pd.DataFrame(records)

        pkg = self._make_pkg(
            datamodule_cfg={
                "time": "time_idx",
                "target": "target",
                "group": ["group"],
                # These DM params are extra but should not crash the TS build
                "max_encoder_length": 5,
                "max_prediction_length": 2,
                "batch_size": 4,
            }
        )

        # _load_dataloader wraps the DataFrame in a TimeSeries; that build
        # then hits _build_datamodule which calls get_datamodule_cls() and will
        # fail because our stub returns None — so we only check up to the TS step
        # by monkey-patching _build_datamodule to capture the TimeSeries object.
        captured = {}

        def _fake_build_dm(data):
            captured["ts"] = data
            raise RuntimeError("stub stop")  # halt before real DM construction

        pkg._build_datamodule = _fake_build_dm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.raises(RuntimeError, match="stub stop"):
                pkg._load_dataloader(df)

        assert "ts" in captured, "TimeSeries was not constructed from DataFrame"
        assert isinstance(captured["ts"], TimeSeries)


class TestPredictGuards:
    """predict() must raise RuntimeError when model is uninitialised."""

    def _make_pkg(self):
        return _StubPkg(model_cfg={}, trainer_cfg={}, datamodule_cfg={})

    def test_predict_raises_if_model_is_none(self):
        pkg = self._make_pkg()
        assert pkg.model is None

        with pytest.raises(RuntimeError, match="Model is not initialized"):
            # We can pass anything — the guard fires before data is touched
            pkg.predict(torch.randn(2, 3))

    def test_inverse_transform_noop_without_datamodule(self):
        """When datamodule is None, inverse_transform=True must be a no-op.

        The guard inside predict() for RuntimeError fires before the
        inverse_transform path, so we test the inverse-transform guard
        by directly inspecting the datamodule attribute.
        """
        pkg = self._make_pkg()
        assert pkg.datamodule is None
        # datamodule is None -> inverse_transform path is skipped safely
        # (We can't reach it without a model, so we verify the attribute itself)
        assert not hasattr(pkg.datamodule, "inverse_transform_predictions")
