# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU coverage for the optional standalone native transport integration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture(name="shim")
def fixture_shim():
    """Load the adapter without loading TE's GPU-dependent package root."""
    path = (
        Path(__file__).resolve().parents[2]
        / "transformer_engine/pytorch/attention/native_cp_transport.py"
    )
    spec = importlib.util.spec_from_file_location("native_cp_shim_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calls_share_the_library_registry(shim, monkeypatch):
    """TE must delegate both initialization and lookup to the same library."""
    parent, logical, transport = object(), object(), object()
    library = SimpleNamespace(
        NativeCPTransport=type("Transport", (), {}),
        initialize_native_cp_transport=Mock(return_value=transport),
        get_native_cp_transport=Mock(return_value=transport),
        set_native_cp_parent_group=Mock(),
        destroy_native_cp_transport=Mock(),
    )
    monkeypatch.setattr(shim.importlib, "import_module", Mock(return_value=library))
    assert shim.NativeCPTransport is library.NativeCPTransport
    assert shim.initialize_native_cp_transport(parent, 4096) is transport
    assert shim.get_native_cp_transport(logical) is transport
    shim.set_native_cp_parent_group(logical, parent)
    shim.destroy_native_cp_transport(parent)
    library.initialize_native_cp_transport.assert_called_once_with(parent, 4096)
    library.get_native_cp_transport.assert_called_once_with(logical)
    library.set_native_cp_parent_group.assert_called_once_with(logical, parent)
    library.destroy_native_cp_transport.assert_called_once_with(parent)


def test_standard_te_does_not_require_optional_library(shim, monkeypatch):
    """Only enabling native CP requires the standalone package."""
    error = ModuleNotFoundError("No module named dcp_transport", name="dcp_transport")
    monkeypatch.setattr(shim.importlib, "import_module", Mock(side_effect=error))
    assert shim.get_native_cp_transport(object()) is None
    shim.destroy_native_cp_transport(object())
    with pytest.raises(ModuleNotFoundError, match="standalone dcp-transport"):
        shim.initialize_native_cp_transport(object(), 4096)


def test_broken_library_dependency_is_not_hidden(shim, monkeypatch):
    """An installed but broken package must not silently disable native CP."""
    error = ModuleNotFoundError(
        "No module named missing_dependency", name="missing_dependency"
    )
    monkeypatch.setattr(shim.importlib, "import_module", Mock(side_effect=error))
    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        shim.get_native_cp_transport(object())
