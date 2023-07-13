from __future__ import annotations

from smash.core._constant import (
    STRUCTURE_PARAMETERS,
    STRUCTURE_STATES,
    STRUCTURE_ADJUST_CI,
    RATIO_PET_HOURLY,
    CSIGN,
    ESIGN,
    ALGORITHM,
    CSIGN_OPTIM,
    ESIGN_OPTIM,
    OPTIM_FUNC,
)

import numpy as np


def test_structure_parameters():
    # % Check parameters gr-a
    assert STRUCTURE_PARAMETERS["gr-a"] == ["cp", "cft", "exc", "lr"]

    # % Check parameters gr-b
    assert STRUCTURE_PARAMETERS["gr-b"] == ["ci", "cp", "cft", "exc", "lr"]

    # % Check parameters gr-c
    assert STRUCTURE_PARAMETERS["gr-c"] == ["ci", "cp", "cft", "cst", "exc", "lr"]

    # % Check parameters gr-d
    assert STRUCTURE_PARAMETERS["gr-d"] == ["cp", "cft", "lr"]

    # % Check parameters vic-a
    assert STRUCTURE_PARAMETERS["vic-a"] == [
        "b",
        "cusl1",
        "cusl2",
        "clsl",
        "ks",
        "ds",
        "dsm",
        "ws",
        "lr",
    ]


def test_structure_states():
    # % Check states gr-a
    assert STRUCTURE_STATES["gr-a"] == ["hp", "hft", "hlr"]

    # % Check states gr-b
    assert STRUCTURE_STATES["gr-b"] == ["hi", "hp", "hft", "hlr"]

    # % Check states gr-c
    assert STRUCTURE_STATES["gr-c"] == ["hi", "hp", "hft", "hst", "hlr"]

    # % Check states gr-d
    assert STRUCTURE_STATES["gr-d"] == ["hp", "hft", "hlr"]

    # % Check states vic-a
    assert STRUCTURE_STATES["vic-a"] == ["husl1", "husl2", "hlsl"]


def test_structure_adjust_ci():
    # % Check adjust ci gr-a
    assert ~STRUCTURE_ADJUST_CI["gr-a"]

    # % Check adjust ci gr-b
    assert STRUCTURE_ADJUST_CI["gr-b"]

    # % Check adjust ci gr-c
    assert STRUCTURE_ADJUST_CI["gr-c"]

    # % Check adjust ci gr-d
    assert ~STRUCTURE_ADJUST_CI["gr-d"]

    # % Check adjust ci vic-a
    assert ~STRUCTURE_ADJUST_CI["vic-a"]


def test_ratio_pet_hourly():
    # % Check ratio_pet_hourly
    assert np.array_equal(
        RATIO_PET_HOURLY,
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.035,
                0.062,
                0.079,
                0.097,
                0.11,
                0.117,
                0.117,
                0.11,
                0.097,
                0.079,
                0.062,
                0.035,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        ),
    )


def test_csign_optim():
    # % Check that all values in CSIGN_OPTIM are in CSIGN
    assert all(sign in CSIGN for sign in CSIGN_OPTIM)


def test_esign_optim():
    # % Check that all values in ESIGN_OPTIM are in ESIGN
    assert all(sign in ESIGN for sign in ESIGN_OPTIM)


def test_optim_fun():
    # % Check that there is a callable for each algorithm
    assert all(alg in ALGORITHM and callable(clb) for alg, clb in OPTIM_FUNC.items())
