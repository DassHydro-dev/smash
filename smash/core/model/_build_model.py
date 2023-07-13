from __future__ import annotations

from smash._constant import (
    STRUCTURE_COMPUTE_CI,
    DEFAULT_OPR_PARAMETERS,
    DEFAULT_OPR_INITIAL_STATES,
)

from smash.core.model._read_input_data import (
    _read_qobs,
    _read_prcp,
    _read_pet,
    _read_descriptor,
)
from smash.core.model._standardize import _standardize_setup

from smash.fcore._mwd_sparse_matrix_manipulation import compute_rowcol_to_ind_sparse
from smash.fcore._mw_atmos_statistic import compute_mean_atmos
from smash.fcore._mw_interception_capacity import compute_interception_capacity

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.fcore._mwd_setup import SetupDT
    from smash.fcore._mwd_mesh import MeshDT
    from smash.fcore._mwd_input_data import Input_DataDT
    from smash.fcore._mwd_parameters import ParametersDT


def _map_dict_to_object(dct: dict, obj: object):
    for key, value in dct.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _build_setup(setup: SetupDT):
    _standardize_setup(setup)

    st = pd.Timestamp(setup.start_time)

    et = pd.Timestamp(setup.end_time)

    setup.ntime_step = (et - st).total_seconds() / setup.dt


def _build_mesh(setup: SetupDT, mesh: MeshDT):
    if setup.sparse_storage:
        compute_rowcol_to_ind_sparse(mesh)  # % Fortran subroutine mw_sparse_storage


def _build_input_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    if setup.read_qobs:
        _read_qobs(setup, mesh, input_data)

    if setup.read_prcp:
        _read_prcp(setup, mesh, input_data)

    if setup.read_pet:
        _read_pet(setup, mesh, input_data)

    if setup.read_descriptor:
        _read_descriptor(setup, mesh, input_data)

    compute_mean_atmos(
        setup, mesh, input_data
    )  # % Fortran subroutine mw_atmos_statistic


def _build_parameters(
    setup: SetupDT,
    mesh: MeshDT,
    input_data: Input_DataDT,
    parameters: ParametersDT,
):
    # % Build parameters
    for key, value in DEFAULT_OPR_PARAMETERS.items():
        if key == "llr":
            setattr(parameters.opr_parameters, key, setup.dt * (value / 3600))
        else:
            setattr(parameters.opr_parameters, key, value)

    # % Build initial states
    for key, value in DEFAULT_OPR_INITIAL_STATES.items():
        setattr(parameters.opr_initial_states, key, value)

    if STRUCTURE_COMPUTE_CI[setup.structure] and setup.dt < 86_400:
        # % Date
        day_index = pd.date_range(
            start=setup.start_time, end=setup.end_time, freq=f"{int(setup.dt)}s"
        )[1:].to_series()

        # % Date to proleptic Gregorian ordinal
        day_index = day_index.apply(lambda x: x.toordinal()).to_numpy()

        # % Scale to 1 (Fortran indexing)
        day_index = day_index - day_index[0] + 1

        compute_interception_capacity(
            setup,
            mesh,
            input_data,
            day_index,
            day_index[-1],
            parameters.opr_parameters.ci,
        )  # % Fortran subroutine mw_interception_capacity