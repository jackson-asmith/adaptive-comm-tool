"""Shared fixtures. No test makes a network call or needs an API key."""

import pytest

from adaptive_comm.personas import load_personas


@pytest.fixture
def personas():
    return load_personas()
