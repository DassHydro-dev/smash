from __future__ import annotations

import warnings
import glob
import os
import errno
import time
from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver.m_setup import SetupDT
    from smash.solver.m_mesh import MeshDT
    from smash.solver.m_input_data import Input_DataDT

from smash.solver.m_utils import sparse_mesh
from smash.core.utils import sparse_matrix_to_vector
from smash.io.raster import read_windowed_raster

import pandas as pd
import numpy as np
import rasterio as rio
import datetime

RATIO_PET_HOURLY = [0, 0, 0, 0, 0, 0, 0, 0.035, 0.062, 0.079, 0.097, 0.11, 0.117, 0.117, 0.11, 0.097, 0.079, 0.062, 0.035,0, 0, 0, 0, 0]

def _derived_type_parser(derived_type, data: dict):

    """
    Derived type parser
    """

    for key, value in data.items():

        if hasattr(derived_type, key):
            setattr(derived_type, key, value)

        else:
            warnings.warn(
                f"'{key}' key does not belong to the derived type {type(derived_type)}",
                UserWarning,
            )


def _standardize_setup(setup: SetupDT):

    """
    Check every SetupDT error/warning exception
    """

    if setup.dt < 0:
        raise ValueError("argument dt of SetupDT is lower than 0")

    if not setup.dt in [900, 3_600, 86_400]:
        warnings.warn(
            "argument dt of SetupDT is not set to a classical value (900, 3600, 86400 seconds)",
            UserWarning,
        )

    if setup.dx < 0:
        raise ValueError("argument dx of SetupDT is lower than 0")

    if setup.start_time.decode() == "...":
        raise ValueError("argument start_time of SetupDT is not defined")

    if setup.end_time.decode() == "...":
        raise ValueError("argument end_time of SetupDT is not defined")

    try:
        st = pd.Timestamp(setup.start_time.decode())
    except:
        raise ValueError("argument start_time of SetupDT is not a valid date")

    try:
        et = pd.Timestamp(setup.end_time.decode())
    except:
        raise ValueError("argument end_time of SetupDT is not a valid date")

    if (et - st).total_seconds() < 0:
        raise ValueError(
            "argument end_time of SetupDT corresponds to an earlier date than start_time"
        )

    if setup.optim_start_time.decode() == "...":
        setup.optim_start_time = setup.start_time
        warnings.warn(
            "argument optim_start_time of SetupDT is not defined. Value set to start_time",
            UserWarning,
        )

    try:
        ost = pd.Timestamp(setup.optim_start_time.decode())
    except:
        raise ValueError("argument optim_start_time of SetupDT is not a valid date")

    if (ost - st).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to an earlier date than start_time"
        )

    if (et - ost).total_seconds() < 0:
        raise ValueError(
            "argument optim_start_time of SetupDT corresponds to a later date than end_time"
        )

    if not setup.active_cell_only and setup.sparse_storage:
        raise ValueError(
            "argument sparse_storage of SetupDT can not be True if active_cell_only of SetupDT is False"
        )

    if setup.active_cell_only and not setup.sparse_storage:
        warnings.warn(
            "argument sparse_storage of SetupDT is False but active_cell_only of SetupDT is True"
        )

    if setup.simulation_only:
        setup.read_qobs = False

    if setup.read_qobs and setup.qobs_directory.decode() == "...":
        raise ValueError(
            "argument simulation_only of SetupDT is False, read_qobs of SetupDT is True and qobs_directory of SetupDT is not defined"
        )

    if setup.read_qobs and not os.path.exists(setup.qobs_directory.decode()):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), setup.qobs_directory.decode()
        )

    if setup.read_prcp and setup.prcp_directory.decode() == "...":
        raise ValueError(
            "argument read_prcp of SetupDT is True and prcp_directory of SetupDT is not defined"
        )

    if setup.read_prcp and not os.path.exists(setup.prcp_directory.decode()):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), setup.prcp_directory.decode()
        )

    if not setup.prcp_format.decode() in ["tiff", "netcdf"]:
        raise ValueError(
            f"argument prpc_format of SetupDT must be one of {['tiff', 'netcdf']} not {setup.prcp_format.decode()}"
        )

    if setup.prcp_conversion_factor < 0:
        raise ValueError("argument prcp_conversion_factor of SetupDT is lower than 0")

    if setup.read_pet and setup.pet_directory.decode() == "...":
        raise ValueError(
            "argument read_pet of SetupDT is True and pet_directory of SetupDT is not defined"
        )

    if setup.read_pet and not os.path.exists(setup.pet_directory.decode()):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), setup.pet_directory.decode()
        )

    if not setup.pet_format.decode() in ["tiff", "netcdf"]:
        raise ValueError(
            f"argument pet_format of SetupDT must be one of {['tiff', 'netcdf']} not {setup.pet_format.decode()}"
        )

    if setup.pet_conversion_factor < 0:
        raise ValueError("argument pet_conversion_factor of SetupDT is lower than 0")

    # TODO, check for better warning/error callbacks


def _standardize_mesh(setup: SetupDT, mesh: MeshDT):

    """
    Check every MeshDT error/warning exception
    """

    if mesh.ncol < 0:
        raise ValueError("argument ncol of MeshDT is lower than 0")

    if mesh.nrow < 0:
        raise ValueError("argument nrow of MeshDT is lower than 0")

    if mesh.ng < 0:
        raise ValueError("argument ng of MeshDT is lower than 0")

    if mesh.xmin < 0:
        raise ValueError("argument xmin of MeshDT is lower than 0")

    if mesh.ymax < 0:
        raise ValueError("argument ymax of MeshDT is lower than 0")

    if np.any(mesh.area < 0):
        raise ValueError(
            "argument area of MeshDT contains at least one value lower than 0"
        )

    if np.all(mesh.flow == -99):
        raise ValueError("argument flow of MeshDT contains only NaN value")

    if np.all(mesh.drained_area == -99):
        raise ValueError("argument drained_area of MeshDT contains only NaN value")

    # TODO add check for remaining MeshDT attributes


def _build_setup(setup: SetupDT):

    """
    Build setup
    """

    _standardize_setup(setup)

    st = pd.Timestamp(setup.start_time.decode())
    ost = pd.Timestamp(setup.optim_start_time.decode())
    et = pd.Timestamp(setup.end_time.decode())

    setup.ntime_step = (et - st).total_seconds() / setup.dt

    setup.optim_start_step = (ost - st).total_seconds() / setup.dt + 1


def _compute_mesh_path(mesh: MeshDT):

    ind = np.unravel_index(
        np.argsort(mesh.drained_area, axis=None), mesh.drained_area.shape
    )

    # Transform from Python to FORTRAN index
    mesh.path[0, :] = ind[0][:] + 1
    mesh.path[1, :] = ind[1][:] + 1


def _build_mesh(setup: SetupDT, mesh: MeshDT):

    """
    Build mesh
    """

    _standardize_mesh(setup, mesh)

    _compute_mesh_path(mesh)
    
    if not setup.active_cell_only:
        
        mesh.global_active_cell = np.where(mesh.flow > 0, 1, mesh.global_active_cell)
        mesh.local_active_cell = mesh.global_active_cell.copy()

    if setup.sparse_storage:

        sparse_mesh(mesh)


def _read_qobs(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    st = pd.Timestamp(setup.start_time.decode())

    code = mesh.code.tobytes(order="F").decode("utf-8").split()

    for i, c in enumerate(code):

        path = glob.glob(
            f"{setup.qobs_directory.decode()}/**/*{c}*.csv", recursive=True
        )

        if len(path) == 0:
            warnings.warn(
                f"No observed discharge file for catchment {c} in recursive root directory {setup.qobs_directory.decode()}"
            )

        elif len(path) > 1:
            raise ValueError(
                f"There is more than one file containing the name of the catchment {c}"
            )

        else:

            with open(path[0], "r") as f:

                header = pd.Timestamp(f.readline())

                time_diff = int((st - header).total_seconds() / setup.dt) + 1

                if time_diff > 0:

                    k = 0

                    for j, line in enumerate(f):

                        if j >= time_diff:

                            try:
                                input_data.qobs[i, k] = float(line)

                                k += 1

                            except:
                                break
                else:

                    k = -time_diff

                    for line in f:

                        try:
                            input_data.qobs[i, k] = float(line)

                            k += 1

                        except:
                            break


def _index_containing_substring(the_list, substring):
    
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1


def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    date_range = pd.date_range(start=setup.start_time.decode(), end=setup.end_time.decode(), freq=f"{int(setup.dt)}s")[1:]
    
    if setup.prcp_format.decode() == "tiff":
        
        files = sorted(glob.glob(f"{setup.prcp_directory.decode()}/**/*tif*", recursive=True))
        
    elif setup.prcp_format.decode() == "netcdf":
        
        files = sorted(glob.glob(f"{setup.prcp_directory.decode()}/**/*nc", recursive=True))
        
    for i, date in enumerate(tqdm(date_range, desc="reading precipitation")):
        
        date_strf = date.strftime("%Y%m%d%H%M")
        
        ind = _index_containing_substring(files, date_strf)
        
        if ind == - 1:
            
            if setup.sparse_storage:
                
                input_data.prcp_sparse[:, i] = -99.
                
            else:
                
                input_data.prcp[..., i] = -99.
            
            warnings.warn(f"Missing precipitation file for date {date}")
        
        else:
            
            matrix = read_windowed_raster(files[ind], mesh) * setup.prcp_conversion_factor
            
            if setup.sparse_storage:
                
                input_data.prcp_sparse[:, i] = sparse_matrix_to_vector(mesh, matrix)
                
            else:
                
                input_data.prcp[..., i] = matrix
        
        files.pop(ind)


def _read_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    
    date_range = pd.date_range(start=setup.start_time.decode(), end=setup.end_time.decode(), freq=f"{int(setup.dt)}s")[1:]
    
    if setup.pet_format.decode() == "tiff":
        
        files = sorted(glob.glob(f"{setup.pet_directory.decode()}/**/*tif*", recursive=True))
        
    elif setup.pet_format.decode() == "netcdf":
        
        files = sorted(glob.glob(f"{setup.pet_directory.decode()}/**/*nc", recursive=True))

    
    if setup.daily_interannual_pet:
        
        leap_year_days = pd.date_range(start="202001010000", end="202012310000", freq="1D")
        nstep_per_day = int(86_400 / setup.dt)
        
        if nstep_per_day == 1:
            
            ratio = [1]
        
        else:
            
            ratio = np.repeat(RATIO_PET_HOURLY, 3_600 / setup.dt) / (3_600 / setup.dt)
        
        for i, day in enumerate(tqdm(leap_year_days, desc="reading daily interannual pet")):
            
            day_strf = day.strftime("%m%d")
            
            ind = _index_containing_substring(files, day_strf)
            
            if ind == - 1:
            
                if setup.sparse_storage:
                    
                    input_data.pet_sparse[:, i] = -99.
                    
                else:
                    
                    input_data.pet[..., i] = -99.
                
                warnings.warn(f"Missing daily interannual pet file for date {date}")
                
            else:
            
                matrix = read_windowed_raster(files[ind], mesh) * setup.pet_conversion_factor
                
                for j in range(nstep_per_day):
                    
                    time = day + j * datetime.timedelta(seconds=setup.dt)
                    
                    ind_time = date_range.indexer_at_time(time)
                    
                    input_data.pet[..., ind_time] = np.repeat(matrix[..., np.newaxis], len(ind_time), axis=2) * ratio[j]

            
            
            
    # ~ else:
        
        # ~ for i, date in enumerate(tqdm(date_range, desc="reading pet")):
        
        # ~ date_strf = date.strftime("%Y%m%d%H%M")
        
        # ~ ind = _index_containing_substring(files, date_strf)
        
        # ~ if ind == - 1:
            
            # ~ if setup.sparse_storage:
                
                # ~ input_data.prcp_sparse[:, i] = -99.
                
            # ~ else:
                
                # ~ input_data.prcp[..., i] = -99.
            
            # ~ warnings.warn(f"Missing pet file for date {date}")
        
        # ~ else:
            
            # ~ matrix = read_windowed_raster(files[ind], mesh) * setup.pet_conversion_factor
            
            # ~ if setup.sparse_storage:
                
                # ~ input_data.pet_sparse[:, i] = sparse_matrix_to_vector(mesh, matrix)
                
            # ~ else:
                
                # ~ input_data.pet[..., i] = matrix
        
        # ~ files.pop(ind)


def _build_input_data(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):

    """
    Build input_data
    """

    if setup.read_qobs:

        _read_qobs(setup, mesh, input_data)

    if setup.read_prcp:

        _read_prcp(setup, mesh, input_data)
        
    if setup.read_pet:
        
        _read_pet(setup, mesh, input_data)
