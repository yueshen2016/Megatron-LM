# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Router storage precision must be independently configurable from compute precision."""

import argparse

import pytest

from examples.post_training.modelopt.export import _export_router_dtype, add_modelopt_export_args


@pytest.mark.parametrize("compute_dtype", [None, "fp32", "fp64"])
def test_default_preserves_existing_router_storage_override(compute_dtype):
    assert _export_router_dtype(None, compute_dtype) == compute_dtype


@pytest.mark.parametrize("compute_dtype", [None, "fp32", "fp64"])
def test_model_storage_uses_export_dtype_instead_of_router_compute_dtype(compute_dtype):
    # Qwen3 MoE computes routing in FP32 but its original checkpoint stores BF16.
    # ModelOpt's None storage override uses the overall export dtype (BF16).
    assert _export_router_dtype("model", compute_dtype) is None


@pytest.mark.parametrize("choice", ["fp32", "fp64"])
def test_explicit_storage_precision_overrides_compute_precision(choice):
    assert _export_router_dtype(choice, None) == choice


@pytest.mark.parametrize("choice", [None, "model", "fp32", "fp64"])
def test_export_storage_cli_choice(choice):
    parser = add_modelopt_export_args(argparse.ArgumentParser())
    argv = [] if choice is None else ["--export-moe-router-dtype", choice]
    assert parser.parse_args(argv).export_moe_router_dtype == choice
