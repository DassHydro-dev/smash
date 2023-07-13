from __future__ import annotations

from smash._constant import SAMPLES_GENERATORS, PROBLEM_KEYS

import warnings

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash._typing import AnyTuple, Numeric


# TODO: Check bounds, parameters name
def _standardize_generate_samples_problem(problem: dict) -> dict:
    if not isinstance(problem, dict):
        raise TypeError("problem argument must be a dictionary")

    if not all(k in problem.keys() for k in PROBLEM_KEYS):
        raise KeyError(
            f"Problem dictionary should be defined with required keys {PROBLEM_KEYS}"
        )

    unk_keys = [k for k in problem.keys() if k not in PROBLEM_KEYS]

    if unk_keys:
        warnings.warn(
            f"Unknown key(s) found in the problem definition {unk_keys}. Choices: {PROBLEM_KEYS}"
        )

    return problem


def _standardize_generate_samples_generator(problem: dict, generator: str) -> str:
    if not isinstance(generator, str):
        raise TypeError("generator argument must be a str")
    generator = generator.lower()

    if generator not in SAMPLES_GENERATORS:
        raise ValueError(
            f"Unknown generator '{generator}': Choices: {SAMPLES_GENERATORS}"
        )

    return generator


def _standardize_generate_samples_n(n: Numeric) -> int:
    if not isinstance(n, (int, float)):
        raise TypeError("n argument must be of Numeric type (int, float)")

    n = int(n)

    if n <= 0:
        raise ValueError("n argument must be greater than 0")

    return n


def _standardize_generate_samples_random_state(random_state: Numeric | None) -> int:
    if random_state is None:
        pass

    else:
        if not isinstance(random_state, (int, float)):
            raise TypeError(
                "random_state argument must be of Numeric type (int, float)"
            )

        random_state = int(random_state)

        if random_state < 0 or random_state > 4_294_967_295:
            raise ValueError("random_state argument must be between 0 and 2**32 - 1")

    return random_state


def _standardize_generate_samples_mean(problem: dict, mean: dict | None) -> dict:
    default_mean = dict(zip(problem["names"], np.mean(problem["bounds"], axis=1)))

    if mean is None:
        mean = default_mean

    else:
        if not isinstance(mean, dict):
            raise TypeError("mean argument must be a dictionary")

        for name, um in mean.items():
            if not name in problem["names"]:
                warnings.warn(
                    f"Key '{name}' does not match any existing names in the problem definition {problem['names']}"
                )

            if isinstance(um, (int, float)):
                default_mean.update({name: um})

            else:
                raise TypeError("mean value must be of Numeric type (int, float)")

        mean = default_mean

    return mean


def _standardize_generate_samples_coef_std(coef_std: Numeric | None) -> float:
    if coef_std is None:
        coef_std = 3.0

    else:
        if not isinstance(coef_std, (int, float)):
            raise TypeError("coef_std argument must be of Numeric type (int, float)")

        coef_stf = float(coef_std)

    return coef_std


def _standardize_generate_samples_args(
    problem: dict,
    generator: str,
    n: Numeric,
    random_state: Numeric | None,
    mean: dict | None,
    coef_std: Numeric | None,
) -> AnyTuple:
    problem = _standardize_generate_samples_problem(problem)

    generator = _standardize_generate_samples_generator(problem, generator)

    n = _standardize_generate_samples_n(n)

    random_state = _standardize_generate_samples_random_state(random_state)

    if generator in ["normal", "gaussian"]:
        mean = _standardize_generate_samples_mean(problem, mean)

        coef_std = _standardize_generate_samples_coef_std(coef_std)

    return (problem, generator, n, random_state, mean, coef_std)
