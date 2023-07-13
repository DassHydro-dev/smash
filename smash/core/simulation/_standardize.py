from __future__ import annotations

from smash.core._constant import (
    ALGORITHM,
    MAPPING,
    GAUGE_ALIAS,
    WGAUGE_ALIAS,
    STRUCTURE_PARAMETERS,
    STRUCTURE_STATES,
    STRUCTURE_ADJUST_CI,
    JOBS_FUN,
    JREG_FUN,
    AUTO_WJREG,
    CSIGN_OPTIM,
    ESIGN_OPTIM,
)

from smash.core._event_segmentation import _standardize_event_seg_options

from smash.solver._mw_derived_type_update import (
    reset_optimize_setup,
    update_optimize_setup_optimize_args,
    update_optimize_setup_optimize_options,
)

from smash.core.generate_samples import (
    _get_bound_constraints,
    generate_samples,
    SampleResult,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model
    from smash.solver._mwd_parameters import ParametersDT
    from smash.solver._mwd_states import StatesDT
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import numpy as np
import pandas as pd
import os
import warnings


def _standardize_mapping(mapping: str) -> str:
    if isinstance(mapping, str):
        mapping = mapping.lower()

    else:
        raise TypeError(f"mapping argument must be str")

    if mapping not in MAPPING:
        raise ValueError(f"Unknown mapping '{mapping}'. Choices: {MAPPING}")

    return mapping


def _standardize_descriptors(input_data: Input_DataDT, setup: SetupDT) -> None:
    # For the moment, return None
    # TODO: add warnings for rejecting uniform descriptors

    if setup._nd == 0:
        raise ValueError(
            f"The optimization method chosen can not be used if no catchment descriptors are available"
        )

    else:
        for i in range(setup._nd):
            d = input_data.descriptor[..., i]

            if np.all(d == d[0, 0]):
                raise ValueError(
                    f"Cannot optimize the Model with spatially uniform descriptor {setup.descriptor_name[i]}"
                )


def _standardize_algorithm(algorithm: str | None, mapping: str) -> str:
    if algorithm is None:
        if mapping == "uniform":
            algorithm = "sbs"

        elif mapping in ["distributed", "hyper-linear", "hyper-polynomial"]:
            algorithm = "l-bfgs-b"

    else:
        if isinstance(algorithm, str):
            algorithm = algorithm.lower()

        else:
            raise TypeError(f"algorithm argument must be str")

        if algorithm not in ALGORITHM:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choices: {ALGORITHM}")

        if algorithm == "sbs":
            if mapping in ["distributed", "hyper-linear", "hyper-polynomial"]:
                raise ValueError(
                    f"'{algorithm}' algorithm can not be used with '{mapping}' mapping"
                )

        elif algorithm == "nelder-mead":
            if mapping in ["distributed", "hyper-polynomial"]:
                raise ValueError(
                    f"'{algorithm}' algorithm can not be used with '{mapping}' mapping"
                )

        elif algorithm == "l-bfgs-b":
            if mapping == "uniform":
                raise ValueError(
                    f"'{algorithm}' algorithm can not be used with '{mapping}' mapping"
                )

    return algorithm


def _standardize_control_vector(
    control_vector: str | list | tuple | None, setup: SetupDT
) -> np.ndarray:
    if control_vector is None:
        control_vector = np.array(STRUCTURE_PARAMETERS[setup.structure])
        if STRUCTURE_ADJUST_CI[setup.structure]:
            if "ci" in control_vector:
                control_vector.remove("ci")
    else:
        if isinstance(control_vector, str):
            control_vector = np.array(control_vector, ndmin=1)

        elif isinstance(control_vector, (list, tuple)):
            control_vector = np.array(control_vector)

        else:
            raise TypeError(f"control_vector argument must be str or list-like object")

        for name in control_vector:
            available = [
                *STRUCTURE_PARAMETERS[setup.structure],
                *STRUCTURE_STATES[setup.structure],
            ]

            if name not in available:
                raise ValueError(
                    f"Unknown parameter or state '{name}' for structure '{setup.structure}' in control_vector. Choices: {available}"
                )

    return control_vector


def _standardize_jobs_fun(jobs_fun: str | list | tuple, algorithm: str) -> np.ndarray:
    if isinstance(jobs_fun, str):
        jobs_fun = np.array(jobs_fun, ndmin=1)

    elif isinstance(jobs_fun, (list, tuple)):
        jobs_fun = np.array(jobs_fun)

    else:
        raise TypeError("jobs_fun argument must be str or list-like object")

    if "kge" in jobs_fun and algorithm == "l-bfgs-b":
        raise ValueError(
            f"'kge' objective function can not be used with '{algorithm}' algorithm (non convex function)"
        )

    list_jobs_fun = JOBS_FUN + CSIGN_OPTIM + ESIGN_OPTIM

    unk_jobs_fun = [jof for jof in jobs_fun if jof not in list_jobs_fun]

    if unk_jobs_fun:
        raise ValueError(
            f"Unknown objective function {unk_jobs_fun}. Choices: {list_jobs_fun}"
        )

    return jobs_fun


def _standardize_wjobs_fun(
    wjobs_fun: list | tuple | None, jobs_fun: np.ndarray, algorithm: str
) -> np.ndarray:
    if wjobs_fun is None:
        # % WIP
        if algorithm == "nsga":
            pass

        else:
            wjobs_fun = np.ones(jobs_fun.size) / jobs_fun.size

    else:
        if isinstance(wjobs_fun, (list, tuple)):
            wjobs_fun = np.array(wjobs_fun)

        else:
            raise TypeError("wjobs_fun argument must list-like object")

        if wjobs_fun.size != jobs_fun.size:
            raise ValueError(
                f"Inconsistent size between jobs_fun ({jobs_fun.size}) and wjobs_fun ({wjobs_fun.size})"
            )

    return wjobs_fun


def _standardize_bounds(
    user_bounds: dict | None,
    control_vector: np.ndarray,
    setup: SetupDT,
    parameters: ParametersDT,
    states: StatesDT,
) -> np.ndarray:
    # % Default values
    bounds = np.empty(shape=(control_vector.size, 2), dtype=np.float32)

    for i, name in enumerate(control_vector):
        if name in setup._parameters_name:
            ind = np.argwhere(setup._parameters_name == name)

            bounds[i, :] = (
                setup._optimize.lb_parameters[ind].item(),
                setup._optimize.ub_parameters[ind].item(),
            )

        elif name in setup._states_name:
            ind = np.argwhere(setup._states_name == name)

            bounds[i, :] = (
                setup._optimize.lb_states[ind].item(),
                setup._optimize.ub_states[ind].item(),
            )

    if user_bounds is None:
        pass

    elif isinstance(user_bounds, dict):
        for name, b in user_bounds.items():
            if name in control_vector:
                if not isinstance(b, (np.ndarray, list, tuple)) or len(b) != 2:
                    raise ValueError(
                        f"bounds values for '{name}' must be list-like object of length 2"
                    )

                if not isinstance(b[0], (int, float)) and b[0] is not None:
                    raise ValueError(
                        f"Lower bound value {b[0]} for '{name}' must be int, float or None"
                    )

                if not isinstance(b[1], (int, float)) and b[1] is not None:
                    raise ValueError(
                        f"Upper bound value {b[1]} for '{name}' must be int, float or None"
                    )

                ind = np.argwhere(control_vector == name)

                if b[0] is not None:
                    bounds[ind, 0] = b[0]

                if b[1] is not None:
                    bounds[ind, 1] = b[1]

                if bounds[ind, 0] >= bounds[ind, 1]:
                    raise ValueError(
                        f"bounds values for '{name}' are invalid, lower bound ({bounds[ind, 0].item()}) is greater than or equal to upper bound ({bounds[ind, 1].item()})"
                    )

            else:
                raise ValueError(
                    f"bounds values for '{name}' cannot be changed because '{name}' is not present in the control vector {control_vector}"
                )

        # % Check that parameters and states are inside user bounds
        for i, name in enumerate(control_vector):
            if name in setup._parameters_name:
                ind = np.argwhere(setup._parameters_name == name)

                parameters_attr = getattr(parameters, name)
                if np.any(parameters_attr + 1e-3 < bounds[i, 0]) or np.any(
                    parameters_attr - 1e-3 > bounds[i, 1]
                ):
                    raise ValueError(
                        f"bounds values for '{name}' are invalid, background parameters [{np.min(parameters_attr)} {np.max(parameters_attr)}] is outside the bounds {bounds[i,:]}"
                    )

            # % Already check, must be states if not parameters
            else:
                ind = np.argwhere(setup._states_name == name)

                states_attr = getattr(states, name)
                if np.any(states_attr + 1e-3 < bounds[i, 0]) or np.any(
                    states_attr - 1e-3 > bounds[i, 1]
                ):
                    raise ValueError(
                        f"bounds values for '{name}' are invalid, background states [{np.min(states_attr)} {np.max(states_attr)}] is outside the bounds {bounds[i,:]}"
                    )

    else:
        raise TypeError(f"bounds argument must be dict")

    return bounds


def _standardize_gauge(
    gauge: str | list | tuple,
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
) -> np.ndarray:
    if isinstance(gauge, str):
        if gauge == "all":
            gauge = mesh.code.copy()

        elif gauge == "downstream":
            ind = np.argmax(mesh.area)

            gauge = np.array(mesh.code[ind], ndmin=1)

        elif gauge in mesh.code:
            gauge = np.array(gauge, ndmin=1)

        else:
            raise ValueError(
                f"Unknown gauge alias or code '{gauge}'. Choices: {GAUGE_ALIAS} or {mesh.code}"
            )

    elif isinstance(gauge, (list, tuple)):
        gauge = np.array(gauge)

    else:
        raise TypeError(f"gauge argument must be str or list-like object")

    gauge_check = np.array([])

    for i, name in enumerate(gauge):
        if name in mesh.code:
            ind = np.argwhere(mesh.code == name).squeeze()

            if np.all(input_data.qobs[ind, setup._optimize.optimize_start_step :] < 0):
                warnings.warn(
                    f"gauge '{name}' has no available observed discharge. Removed from the optimization"
                )

            else:
                gauge_check = np.append(gauge_check, gauge[i])

        else:
            raise ValueError(f"Unknown gauge code '{name}'. Choices: {mesh.code}")

    if gauge_check.size == 0:
        raise ValueError(
            f"No available observed discharge for optimization at gauge(s) {gauge}"
        )

    else:
        gauge = gauge_check

    return gauge


def _standardize_wgauge(
    wgauge: str | list | tuple, gauge: np.ndarray, mesh: MeshDT
) -> np.ndarray:
    weight_arr = np.zeros(shape=mesh.code.size, dtype=np.float32)
    ind = np.in1d(mesh.code, gauge)

    if isinstance(wgauge, str):
        if wgauge == "mean":
            weight_arr[ind] = 1 / gauge.size

        elif wgauge == "median":
            weight_arr[ind] = -50

        elif wgauge == "area":
            weight_arr[ind] = mesh.area[ind] / sum(mesh.area[ind])

        elif wgauge == "minv_area":
            weight_arr[ind] = (1 / mesh.area[ind]) / sum(1 / mesh.area[ind])

        else:
            raise ValueError(
                f"Unknown wgauge alias '{wgauge}'. Choices: {WGAUGE_ALIAS}"
            )

    elif isinstance(wgauge, (list, tuple)):
        wgauge = np.array(wgauge)

        if wgauge.size != gauge.size:
            raise ValueError(
                f"Inconsistent size between gauge ({gauge.size}) and wgauge ({wgauge.size})"
            )

        elif np.any(wgauge < 0):
            raise ValueError(f"wgauge can not receive negative values ({wgauge})")

        else:
            weight_arr[ind] = wgauge

    else:
        raise TypeError(f"wgauge argument must be str or list-like object")

    wgauge = weight_arr

    return wgauge


def _standardize_ost(ost: str | pd.Timestamp | None, setup: SetupDT) -> pd.Timestamp:
    if ost is None:
        ost = pd.Timestamp(setup.start_time)

    else:
        st = pd.Timestamp(setup.start_time)
        et = pd.Timestamp(setup.end_time)

        if isinstance(ost, str):
            try:
                ost = pd.Timestamp(ost)

            except:
                raise ValueError(f"ost '{ost}' argument is an invalid date")

        elif isinstance(ost, pd.Timestamp):
            pass

        else:
            raise TypeError(f"ost argument must be str or pandas.Timestamp object")

        if (ost - st).total_seconds() < 0 or (et - ost).total_seconds() < 0:
            raise ValueError(
                f"ost '{ost}' argument must be between start time '{st}' and end time '{et}'"
            )

    return ost


def _standardize_ncpu(ncpu: int) -> int:
    if ncpu < 1:
        raise ValueError("ncpu argument must be greater or equal to 1")

    elif ncpu > os.cpu_count():
        warnings.warn(
            f"ncpu argument is greater than the total number of CPUs in the system. ncpu is set to {os.cpu_count()}"
        )
        ncpu = os.cpu_count()

    return ncpu


def _standardize_maxiter(maxiter: int) -> int:
    if isinstance(maxiter, int):
        if maxiter < 0:
            raise ValueError("maxiter option must be greater or equal to 0")

    else:
        raise TypeError("maxiter option must be integer")

    return maxiter


def _standardize_jreg_fun(jreg_fun: str | list | tuple, setup: SetupDT) -> np.ndarray:
    if isinstance(jreg_fun, str):
        jreg_fun = np.array(jreg_fun, ndmin=1)

    elif isinstance(jreg_fun, (list, tuple)):
        jreg_fun = np.array(jreg_fun)

    else:
        raise TypeError("jreg_fun option must str or list-like object")

    if list(jreg_fun) and setup._optimize.mapping.startswith("hyper"):
        # re-check if future jreg_fun can be handled

        raise ValueError(
            f"Regularization function(s) can not be used with '{setup._optimize.mapping}' mapping"
        )

    unk_jreg_fun = [jrf for jrf in jreg_fun if jrf not in JREG_FUN]

    if unk_jreg_fun:
        raise ValueError(
            f"Unknown regularization function(s) {unk_jreg_fun}. Choices: {JREG_FUN}"
        )

    return jreg_fun


def _standardize_wjreg(wjreg: int | float, jreg_fun: np.ndarray) -> float:
    if isinstance(wjreg, (int, float)):
        if wjreg < 0:
            raise ValueError("wjreg option must be greater or equal to 0")

        if jreg_fun.size == 0:
            warnings.warn(
                "No regularization function has been choosen with the options jreg_fun. wjreg option will have no effect"
            )

    else:
        raise TypeError("wjreg option must be integer or float")

    return wjreg


def _standardize_wjreg_fun(wjreg_fun: list | tuple, jreg_fun: np.ndarray) -> np.ndarray:
    if isinstance(wjreg_fun, (list, tuple)):
        wjreg_fun = np.array(wjreg_fun)

    else:
        raise TypeError("wjreg_fun option must be a list-like object")

    if wjreg_fun.size != jreg_fun.size:
        raise ValueError(
            f"Inconsistent size between jreg_fun ({jreg_fun.size}) and wjreg_fun ({wjreg_fun.size})"
        )

    return wjreg_fun


def _standardize_auto_wjreg(auto_wjreg: str, jreg_fun: np.ndarray) -> str:
    if isinstance(auto_wjreg, str):
        if auto_wjreg not in AUTO_WJREG:
            raise ValueError(
                f"Unknown auto_wjreg '{auto_wjreg}'. Choices: {AUTO_WJREG}"
            )

        if jreg_fun.size == 0:
            warnings.warn(
                "No regularization function has been choosen with the option jreg_fun. auto_wjreg option will have no effect"
            )

    else:
        raise TypeError("auto_wjreg option must be str")

    return auto_wjreg


def _standardize_nb_wjreg_lcurve(
    nb_wjreg_lcurve: int, auto_wjreg: str, jreg_fun: np.ndarray
) -> str:
    if isinstance(nb_wjreg_lcurve, int):
        if nb_wjreg_lcurve < 5:
            raise ValueError("nb_wjreg_lcurve option must be greater or equal to 6")

        if auto_wjreg != "lcurve":
            warnings.warn(
                "auto_wjreg option is not set to 'lcurve'. nb_wjreg_lcurve option will have no effect"
            )

        if jreg_fun.size == 0:
            warnings.warn(
                "No regularization function has been choosen with the option jreg_fun. nb_wjreg_lcurve option will have no effect"
            )

    else:
        raise TypeError("nb_wjreg_lcurve option must be int")

    return nb_wjreg_lcurve


# %TODO: Add "distance_correlation" once clearly verified
def standardize_reg_descriptors(reg_descriptors: dict, setup: SetupDT) -> dict:
    reg_descriptors_for_params = np.zeros(
        shape=(setup._parameters_name.size, setup._nd), dtype=int
    )
    reg_descriptors_for_states = np.zeros(
        shape=(setup._states_name.size, setup._nd), dtype=int
    )

    standardized_reg_descriptors = {}

    if isinstance(reg_descriptors, dict):
        for p, desc in reg_descriptors.items():
            if p in setup._parameters_name:
                ind = np.argwhere(setup._parameters_name == p)

                if isinstance(desc, str):
                    pos = np.argwhere(setup.descriptor_name == desc)
                    reg_descriptors_for_params[ind, 0] = pos + 1

                elif isinstance(desc, list):
                    i = 0
                    for d in desc:
                        pos = np.argwhere(setup.descriptor_name == d)
                        reg_descriptors_for_params[ind, i] = pos + 1
                        i = i + 1

                else:
                    raise ValueError(
                        f"reg_descriptors['{p}'], '{desc}', must be a string or list"
                    )

            if p in setup._states_name:
                ind = np.argwhere(setup._states_name == p)

                if isinstance(desc, str):
                    pos = np.argwhere(setup.descriptor_name == desc)
                    reg_descriptors_for_states[ind, 0] = pos + 1

                elif isinstance(desc, list):
                    i = 0
                    for d in desc:
                        pos = np.argwhere(setup.descriptor_name == d)
                        reg_descriptors_for_states[ind, i] = pos + 1
                        i = i + 1

                else:
                    raise ValueError(
                        f"reg_descriptors['{p}'], '{desc}', must be a string or list"
                    )

        standardized_reg_descriptors = {
            "params": reg_descriptors_for_params,
            "states": reg_descriptors_for_states,
        }

    else:
        raise ValueError(
            f"reg_descriptors '{reg_descriptors}' argument must be a dictionary"
        )

    return standardized_reg_descriptors


def _standardize_adjoint_test(adjoint_test: bool, setup: SetupDT) -> bool:
    if isinstance(adjoint_test, bool):
        if adjoint_test:
            if setup._optimize.mapping.startswith("hyper"):
                warnings.warn(
                    f"adjoint test option is not yet available with '{setup._optimize.mapping}' mapping"
                )

    else:
        raise TypeError("adjoint_test option must be bool")

    return adjoint_test


def _standardize_return_lcurve(return_lcurve: bool, auto_wjreg: str) -> bool:
    if isinstance(return_lcurve, bool):
        if return_lcurve and auto_wjreg != "lcurve":
            raise ValueError(
                "return_lcurve option can only be used with auto_wjreg option set to 'lcurve'"
            )

    else:
        raise TypeError("return_lcurve option must be bool")

    return return_lcurve


def _standardize_jobs_fun_wo_mapping(
    jobs_fun: str | list | tuple,
) -> np.ndarray:
    if isinstance(jobs_fun, str):
        jobs_fun = np.array(jobs_fun, ndmin=1)

    elif isinstance(jobs_fun, (list, tuple)):
        jobs_fun = np.array(jobs_fun)

    else:
        raise TypeError("jobs_fun argument must be str or list-like object")

    list_jobs_fun = JOBS_FUN + CSIGN_OPTIM + ESIGN_OPTIM

    unk_jobs_fun = [jof for jof in jobs_fun if jof not in list_jobs_fun]

    if unk_jobs_fun:
        raise ValueError(
            f"Unknown objective function(s) {unk_jobs_fun}. Choices: {list_jobs_fun}"
        )

    return jobs_fun


def _standardize_wjobs_fun_wo_mapping(
    wjobs_fun: list | None, jobs_fun: np.ndarray
) -> np.ndarray:
    if wjobs_fun is None:
        wjobs_fun = np.ones(jobs_fun.size) / jobs_fun.size

    else:
        if isinstance(wjobs_fun, (list, tuple)):
            wjobs_fun = np.array(wjobs_fun)

        else:
            raise TypeError("wjobs_fun argument must list-like object")

        if wjobs_fun.size != jobs_fun.size:
            raise ValueError(
                f"Inconsistent size between jobs_fun ({jobs_fun.size}) and wjobs_fun ({wjobs_fun.size})"
            )

    return wjobs_fun


def _standardize_bayes_estimate_sample(
    sample: SampleResult | None,
    n: int,
    random_state: int | None,
    setup: SetupDT,
):
    if sample is None:
        sample = generate_samples(
            problem=_get_bound_constraints(setup, states=False),
            generator="uniform",
            n=n,
            random_state=random_state,
        )

    elif isinstance(sample, SampleResult):
        param_state = (
            STRUCTURE_PARAMETERS[setup.structure] + STRUCTURE_STATES[setup.structure]
        )

        unk_cv = [cv for cv in sample._problem["names"] if cv not in param_state]

        if unk_cv:
            warnings.warn(
                f"Problem names ({unk_cv}) do not present for any Model parameters and/or states in the {setup.structure} structure"
            )

    else:
        raise TypeError("sample must be a SampleResult object or None")

    return sample


def _standardize_multiple_run_sample(
    sample: SampleResult,
    setup: SetupDT,
):
    if isinstance(sample, SampleResult):
        param_state = (
            STRUCTURE_PARAMETERS[setup.structure] + STRUCTURE_STATES[setup.structure]
        )

        unk_cv = [cv for cv in sample._problem["names"] if cv not in param_state]

        if unk_cv:
            warnings.warn(
                f"Problem names ({unk_cv}) do not present for any Model parameters and/or states in the {setup.structure} structure"
            )

    else:
        raise TypeError("sample must be a SampleResult object")

    return sample


def _standardize_bayes_optimize_sample(
    sample: SampleResult | None,
    n: int,
    random_state: int | None,
    control_vector: np.ndarray,
    bounds: np.ndarray,
):
    if sample is None:
        problem = {
            "num_vars": len(control_vector),
            "names": list(control_vector),
            "bounds": [list(bound) for bound in bounds],
        }

        sample = generate_samples(
            problem=problem,
            generator="uniform",
            n=n,
            random_state=random_state,
        )

    elif isinstance(sample, SampleResult):
        # check if problem names and control_vector have the same elements
        if set(sample._problem["names"]) != set(control_vector):
            raise ValueError(
                f"Problem names ({sample._problem['names']}) and control vectors ({control_vector}) must have the same elements"
            )

        else:  # check if problem bounds inside optimization bounds
            dict_prl = dict(zip(sample._problem["names"], sample._problem["bounds"]))

            dict_optim = dict(zip(control_vector, bounds))

            for key, val in dict_optim.items():
                if val[0] > dict_prl[key][0] or val[1] < dict_prl[key][1]:
                    raise ValueError(
                        f"Problem bound of {key} ({dict_prl[key]}) is outside the boundary condition {val}"
                    )

    else:
        raise TypeError("sample must be a SampleResult object or None")

    return sample


def _standardize_bayes_alpha(
    alpha: int | float | range | list | tuple | np.ndarray,
):
    if isinstance(alpha, (int, float, list)):
        pass

    elif isinstance(alpha, (range, np.ndarray, tuple)):
        alpha = list(alpha)

    else:
        raise TypeError("alpha must be numerical or list-like object")

    return alpha


def _standardize_multiple_run_args(
    sample: SampleResult,
    jobs_fun: str | list | tuple,
    wjobs_fun: list | None,
    event_seg: dict | None,
    gauge: str | list | tuple,
    wgauge: str | list | tuple,
    ost: str | pd.Timestamp | None,
    ncpu: int,
    instance: Model,
):
    reset_optimize_setup(
        instance.setup._optimize
    )  # % Fortran subroutine mw_derived_type_update

    sample = _standardize_multiple_run_sample(sample, instance.setup)

    jobs_fun = _standardize_jobs_fun_wo_mapping(jobs_fun)

    wjobs_fun = _standardize_wjobs_fun_wo_mapping(wjobs_fun, jobs_fun)

    event_seg = _standardize_event_seg_options(event_seg)

    gauge = _standardize_gauge(
        gauge, instance.setup, instance.mesh, instance.input_data
    )

    wgauge = _standardize_wgauge(wgauge, gauge, instance.mesh)

    ost = _standardize_ost(ost, instance.setup)

    ncpu = _standardize_ncpu(ncpu)

    update_optimize_setup_optimize_args(
        instance.setup._optimize,
        "...",
        instance.setup._ntime_step,
        instance.setup._nd,
        instance.mesh.ng,
        jobs_fun.size,
    )  # % Fortran subroutine mw_derived_type_update

    return (
        sample,
        jobs_fun,
        wjobs_fun,
        event_seg,
        wgauge,
        ost,
        ncpu,
    )


def _standardize_optimize_args(
    mapping: str,
    algorithm: str | None,
    control_vector: str | list | tuple | None,
    jobs_fun: str | list | tuple,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | None,
    gauge: str | list | tuple,
    wgauge: str | list | tuple,
    ost: str | pd.Timestamp | None,
    instance: Model,
):
    reset_optimize_setup(
        instance.setup._optimize
    )  # % Fortran subroutine mw_derived_type_update

    mapping = _standardize_mapping(mapping)

    if mapping.startswith("hyper"):
        _standardize_descriptors(instance.input_data, instance.setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, instance.setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    wjobs_fun = _standardize_wjobs_fun(wjobs_fun, jobs_fun, algorithm)

    event_seg = _standardize_event_seg_options(event_seg)

    bounds = _standardize_bounds(
        bounds, control_vector, instance.setup, instance.parameters, instance.states
    )

    gauge = _standardize_gauge(
        gauge, instance.setup, instance.mesh, instance.input_data
    )

    wgauge = _standardize_wgauge(wgauge, gauge, instance.mesh)

    ost = _standardize_ost(ost, instance.setup)

    update_optimize_setup_optimize_args(
        instance.setup._optimize,
        mapping,
        instance.setup._ntime_step,
        instance.setup._nd,
        instance.mesh.ng,
        jobs_fun.size,
    )  # % Fortran subroutine mw_derived_type_update

    return (
        mapping,
        algorithm,
        control_vector,
        jobs_fun,
        wjobs_fun,
        event_seg,
        bounds,
        wgauge,
        ost,
    )


def _standardize_bayes_estimate_args(
    sample: SampleResult,
    alpha: int | float | range | list | tuple | np.ndarray,
    n: int,
    random_state: int | None,
    jobs_fun: str | list | tuple,
    wjobs_fun: list | None,
    event_seg: dict | None,
    gauge: str | list | tuple,
    wgauge: str | list | tuple,
    ost: str | pd.Timestamp | None,
    ncpu: int,
    instance: Model,
):
    reset_optimize_setup(
        instance.setup._optimize
    )  # % Fortran subroutine mw_derived_type_update

    sample = _standardize_bayes_estimate_sample(sample, n, random_state, instance.setup)

    alpha = _standardize_bayes_alpha(alpha)

    jobs_fun = _standardize_jobs_fun_wo_mapping(jobs_fun)

    wjobs_fun = _standardize_wjobs_fun_wo_mapping(wjobs_fun, jobs_fun)

    event_seg = _standardize_event_seg_options(event_seg)

    gauge = _standardize_gauge(
        gauge, instance.setup, instance.mesh, instance.input_data
    )

    wgauge = _standardize_wgauge(wgauge, gauge, instance.mesh)

    ost = _standardize_ost(ost, instance.setup)

    ncpu = _standardize_ncpu(ncpu)

    update_optimize_setup_optimize_args(
        instance.setup._optimize,
        "...",
        instance.setup._ntime_step,
        instance.setup._nd,
        instance.mesh.ng,
        jobs_fun.size,
    )  # % Fortran subroutine mw_derived_type_update

    return (
        sample,
        alpha,
        jobs_fun,
        wjobs_fun,
        event_seg,
        wgauge,
        ost,
        ncpu,
    )


def _standardize_bayes_optimize_args(
    sample: SampleResult,
    alpha: int | float | range | list | tuple | np.ndarray,
    n: int,
    random_state: int | None,
    mapping: str,
    algorithm: str | None,
    control_vector: str | list | tuple | None,
    jobs_fun: str | list | tuple,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | None,
    gauge: str | list | tuple,
    wgauge: str | list | tuple,
    ost: str | pd.Timestamp | None,
    ncpu: int,
    instance: Model,
):
    reset_optimize_setup(
        instance.setup._optimize
    )  # % Fortran subroutine mw_derived_type_update

    mapping = _standardize_mapping(mapping)

    if mapping.startswith("hyper"):
        _standardize_descriptors(instance.input_data, instance.setup)

    algorithm = _standardize_algorithm(algorithm, mapping)

    control_vector = _standardize_control_vector(control_vector, instance.setup)

    jobs_fun = _standardize_jobs_fun(jobs_fun, algorithm)

    wjobs_fun = _standardize_wjobs_fun(wjobs_fun, jobs_fun, algorithm)

    event_seg = _standardize_event_seg_options(event_seg)

    bounds = _standardize_bounds(
        bounds, control_vector, instance.setup, instance.parameters, instance.states
    )

    gauge = _standardize_gauge(
        gauge, instance.setup, instance.mesh, instance.input_data
    )

    wgauge = _standardize_wgauge(wgauge, gauge, instance.mesh)

    ost = _standardize_ost(ost, instance.setup)

    ncpu = _standardize_ncpu(ncpu)

    sample = _standardize_bayes_optimize_sample(
        sample, n, random_state, control_vector, bounds
    )

    alpha = _standardize_bayes_alpha(alpha)

    update_optimize_setup_optimize_args(
        instance.setup._optimize,
        mapping,
        instance.setup._ntime_step,
        instance.setup._nd,
        instance.mesh.ng,
        jobs_fun.size,
    )  # % Fortran subroutine mw_derived_type_update

    return (
        sample,
        alpha,
        mapping,
        algorithm,
        jobs_fun,
        wjobs_fun,
        event_seg,
        bounds,
        wgauge,
        ost,
        ncpu,
    )


def _standardize_ann_optimize_args(
    control_vector: str | list | tuple | None,
    jobs_fun: str | list | tuple,
    wjobs_fun: list | None,
    event_seg: dict | None,
    bounds: list | tuple | None,
    gauge: str | list | tuple,
    wgauge: str | list | tuple,
    ost: str | pd.Timestamp | None,
    instance: Model,
):
    reset_optimize_setup(
        instance.setup._optimize
    )  # % Fortran subroutine mw_derived_type_update

    _standardize_descriptors(instance.input_data, instance.setup)

    control_vector = _standardize_control_vector(control_vector, instance.setup)

    jobs_fun = _standardize_jobs_fun_wo_mapping(jobs_fun)

    wjobs_fun = _standardize_wjobs_fun_wo_mapping(wjobs_fun, jobs_fun)

    event_seg = _standardize_event_seg_options(event_seg)

    bounds = _standardize_bounds(
        bounds, control_vector, instance.setup, instance.parameters, instance.states
    )

    gauge = _standardize_gauge(
        gauge, instance.setup, instance.mesh, instance.input_data
    )

    wgauge = _standardize_wgauge(wgauge, gauge, instance.mesh)

    ost = _standardize_ost(ost, instance.setup)

    update_optimize_setup_optimize_args(
        instance.setup._optimize,
        "...",
        instance.setup._ntime_step,
        instance.setup._nd,
        instance.mesh.ng,
        jobs_fun.size,
    )  # % Fortran subroutine mw_derived_type_update

    return control_vector, jobs_fun, wjobs_fun, event_seg, bounds, wgauge, ost


def _standardize_optimize_options(options: dict | None, instance: Model) -> dict:
    if options is None:
        options = {}
    else:
        if isinstance(options, dict):
            if "maxiter" in options.keys():
                options.update({"maxiter": _standardize_maxiter(options["maxiter"])})

            if "jreg_fun" in options.keys():
                options.update(
                    {
                        "jreg_fun": _standardize_jreg_fun(
                            options["jreg_fun"], instance.setup
                        )
                    }
                )

            if "wjreg" in options.keys():
                options.update(
                    {
                        "wjreg": _standardize_wjreg(
                            options["wjreg"], options.get("jreg_fun", np.empty(shape=0))
                        )
                    }
                )

            if "wjreg_fun" in options.keys():
                options.update(
                    {
                        "wjreg_fun": _standardize_wjreg_fun(
                            options["wjreg_fun"],
                            options.get("jreg_fun", np.empty(shape=0)),
                        )
                    }
                )

            if "auto_wjreg" in options.keys():
                options.update(
                    {
                        "auto_wjreg": _standardize_auto_wjreg(
                            options["auto_wjreg"],
                            options.get("jreg_fun", np.empty(shape=0)),
                        )
                    }
                )

            if "nb_wjreg_lcurve" in options.keys():
                options.update(
                    {
                        "nb_wjreg_lcurve": _standardize_nb_wjreg_lcurve(
                            options["nb_wjreg_lcurve"],
                            options.get("auto_wjreg", None),
                            options.get("jreg_fun", np.empty(shape=0)),
                        )
                    }
                )

            if "adjoint_test" in options.keys():
                options.update(
                    {
                        "adjoint_test": _standardize_adjoint_test(
                            options["adjoint_test"], instance.setup
                        )
                    }
                )

            if "return_lcurve" in options.keys():
                options.update(
                    {
                        "return_lcurve": _standardize_return_lcurve(
                            options["return_lcurve"], options.get("auto_wjreg", None)
                        )
                    }
                )

            update_optimize_setup_optimize_options(
                instance.setup._optimize,
                options.get("jreg_fun", np.empty(shape=0)).size,
            )  # % Fortran subroutine mw_derived_type_update

        else:
            raise TypeError(f"options argument must be a dict")

    return options
