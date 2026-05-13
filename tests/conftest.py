"""Shared pytest configuration for the test suite."""

import gc

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a CUDA GPU")


@pytest.fixture(autouse=True)
def _cleanup_cuda_after_test():
    """Free GPU memory after each test to prevent OOM from accumulation."""
    yield
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
