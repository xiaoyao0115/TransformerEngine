# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compatibility entry points for the independent :mod:`dcp_transport` library.

All native arenas and group aliases belong to dcp_transport. Standard TE use
without native CP remains available when the optional library is not installed.
"""

import importlib


def _library(required=True):
    """Load the optional native transport library."""
    try:
        return importlib.import_module("dcp_transport")
    except ModuleNotFoundError as error:
        if error.name != "dcp_transport":
            raise
        if required:
            raise ModuleNotFoundError(
                "Native CP requires the standalone dcp-transport package. "
                "Install it in the same environment as Transformer Engine."
            ) from error
        return None


def __getattr__(name):
    """Preserve the former NativeCPTransport class import."""
    if name == "NativeCPTransport":
        return _library().NativeCPTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def initialize_native_cp_transport(parent_group, payload_bytes):
    """Collectively initialize the shared library's parent transport."""
    return _library().initialize_native_cp_transport(parent_group, payload_bytes)


def set_native_cp_parent_group(cp_group, parent_group):
    """Map a logical CP group to the shared parent transport."""
    _library().set_native_cp_parent_group(cp_group, parent_group)


def get_native_cp_transport(group):
    """Return the shared transport, or None when native CP is unused."""
    library = _library(required=False)
    return None if library is None else library.get_native_cp_transport(group)


def destroy_native_cp_transport(parent_group):
    """Collectively release the shared transport and its aliases."""
    library = _library(required=False)
    if library is not None:
        library.destroy_native_cp_transport(parent_group)
