from __future__ import annotations

import smash

import numpy as np
import pytest


def generic_forward_run(model: smash.Model, **kwargs) -> dict:
    instance = smash.forward_run(model)

    res = {"forward_run.qsim": instance.sim_response.q[:].flatten()}

    return res


def test_run():
    res = generic_forward_run(pytest.model)

    for key, value in res.items():
        # % Check cost in run
        assert np.allclose(value, pytest.baseline[key][:], atol=1e-06), key
