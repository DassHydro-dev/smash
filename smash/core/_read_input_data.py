from __future__ import annotations

from smash.tools.raster_handler import gdal_read_windowed_raster

from smash.core.utils import sparse_matrix_to_vector

from smash.core._constant import RATIO_PET_HOURLY

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.solver._mwd_setup import SetupDT
    from smash.solver._mwd_mesh import MeshDT
    from smash.solver._mwd_input_data import Input_DataDT

import os
import warnings
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime


def _read_qobs(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    st = pd.Timestamp(setup.start_time)

    for i, c in enumerate(mesh.code):
        path = glob.glob(f"{setup.qobs_directory}/**/*{c}*.csv", recursive=True)

        if len(path) == 0:
            warnings.warn(
                f"No observed discharge file for catchment {c} in recursive root directory {setup.qobs_directory}"
            )

        elif len(path) > 1:
            raise ValueError(
                f"There is more than one file containing the name of the catchment {c}"
            )

        else:
            with open(path[0], "r") as f:
                
                try:
                    
                    header_string=f.readline()
                    header = pd.Timestamp(header_string)
                    
                except:
                    
                    raise ValueError(
                        f"Bad header {header_string} string when reading file '{path[0]}'. '{header_string}' may not be a date."
                    )
                
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


# % Adjust left files (sorted by date - only works if files have the same name)
def _adjust_left_files(files: list[str], date_range: pd.Timestamp):
    n = 0
    ind = -1
    while ind == -1:
        ind = _index_containing_substring(files, date_range[n].strftime("%Y%m%d%H%M"))

        n += 1

    return files[ind:]


def _index_containing_substring(the_list: list, substring: str):
    for i, s in enumerate(the_list):
        if substring in s:
            return i
    return -1


def _split_date(date_datetime):
    
    date_strf = date_datetime.strftime("%Y%m%d%H%M")
    
    year=date_strf[0:4]
    month=date_strf[4:6]
    day=date_strf[6:8]
    
    return year,month,day


def _list_prcp_file(setup):
    
    datetime_date_start=datetime.datetime.fromisoformat(setup.start_time)
    datetime_date_end=datetime.datetime.fromisoformat(setup.end_time)
    
    if datetime_date_end<datetime_date_start:
        raise ValueError(
            f"Cannot list precipitation file because date_end<date_start."
        )
    
    list_file=list()
    
    s_year,s_month,s_day=_split_date(datetime_date_start)
    e_year,e_month,e_day=_split_date(datetime_date_end)
    
    if int(e_year)>int(s_year):
        
        for yy in range(int(s_year),int(e_year)+1):
            list_file=list_file+(glob.glob(f"{setup.prcp_directory}/{yy:04n}/**/*{setup.prcp_format}*", recursive=True))
    
    elif int(e_month)>int(s_month):
        
        for mm in range(int(s_month),int(e_month)+1):
            
            list_file=list_file+(glob.glob(f"{setup.prcp_directory}/{s_year}/{mm:02n}/**/*{setup.prcp_format}*", recursive=True))
        
    elif int(e_day)>int(s_day):
        
        for dd in range(int(s_day),int(e_day)+1):
            list_file=list_file+(glob.glob(f"{setup.prcp_directory}/{s_year}/{s_month}/{dd:02n}/*{setup.prcp_format}*", recursive=True))
    
    else:
        
        list_file=list_file+(glob.glob(f"{setup.prcp_directory}/{s_year}/{s_month}/{s_day}/*{setup.prcp_format}*", recursive=True))
    
    return sorted(list_file)


def _read_prcp(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.prcp_yyyymmdd_access:
        
        files=_list_prcp_file(setup)
        
        if setup.prcp_format == "tif":
            files = _adjust_left_files(files, date_range)
        
    else :
        
        if setup.prcp_format == "tif":
            files = sorted(glob.glob(f"{setup.prcp_directory}/**/*tif*", recursive=True))

            files = _adjust_left_files(files, date_range)

        # % WIP
        elif setup.prcp_format == "nc":
            files = sorted(glob.glob(f"{setup.prcp_directory}/**/*nc", recursive=True))

    for i, date in enumerate(tqdm(date_range, desc="</> Reading precipitation")):
        date_strf = date.strftime("%Y%m%d%H%M")
            
        ind = _index_containing_substring(files, date_strf)
        
        if ind == -1:
            if setup.sparse_storage:
                input_data.sparse_prcp[:, i] = -99.0

            else:
                input_data.prcp[..., i] = -99.0

            warnings.warn(f"Missing precipitation file for date {date}")

        else:
            matrix = (
                gdal_read_windowed_raster(
                    filename=files[ind], smash_mesh=mesh, band=1, lacuna=-99.0
                )
                * setup.prcp_conversion_factor
            )

            if setup.sparse_storage:
                input_data.sparse_prcp[:, i] = sparse_matrix_to_vector(mesh, matrix)

            else:
                input_data.prcp[..., i] = matrix

            files.pop(ind)


def _read_pet(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    date_range = pd.date_range(
        start=setup.start_time,
        end=setup.end_time,
        freq=f"{int(setup.dt)}s",
    )[1:]

    if setup.pet_format == "tif":
        files = sorted(glob.glob(f"{setup.pet_directory}/**/*tif*", recursive=True))

        if not setup.daily_interannual_pet:
            files = _adjust_left_files(files, date_range)

    elif setup.pet_format == "nc":
        files = sorted(glob.glob(f"{setup.pet_directory}/**/*nc", recursive=True))

    if setup.daily_interannual_pet:
        leap_year_days = pd.date_range(
            start="202001010000", end="202012310000", freq="1D"
        )
        nstep_per_day = int(86_400 / setup.dt)
        hourly_ratio = 3_600 / setup.dt

        if hourly_ratio >= 1:
            ratio = np.repeat(RATIO_PET_HOURLY, hourly_ratio) / hourly_ratio

        else:
            ratio = np.sum(RATIO_PET_HOURLY.reshape(-1, int(1 / hourly_ratio)), axis=1)

        for i, day in enumerate(
            tqdm(leap_year_days, desc="</> Reading daily interannual pet")
        ):
            if day.day_of_year in date_range.day_of_year:
                day_strf = day.strftime("%m%d")

                ind = _index_containing_substring(files, day_strf)

                ind_day = np.where(day.day_of_year == date_range.day_of_year)

                if ind == -1:
                    if setup.sparse_storage:
                        input_data.sparse_pet[:, ind_day] = -99.0

                    else:
                        input_data.pet[..., ind_day] = -99.0

                    warnings.warn(
                        f"Missing daily interannual pet file for date {day_strf}"
                    )

                else:
                    subset_date_range = date_range[ind_day]

                    matrix = (
                        gdal_read_windowed_raster(
                            filename=files[ind], smash_mesh=mesh, band=1, lacuna=-99.0
                        )
                        * setup.pet_conversion_factor
                    )

                    if setup.sparse_storage:
                        vector = sparse_matrix_to_vector(mesh, matrix)

                    for j in range(nstep_per_day):
                        step = day + j * datetime.timedelta(seconds=setup.dt)

                        ind_step = subset_date_range.indexer_at_time(step)

                        if setup.sparse_storage:
                            input_data.sparse_pet[:, ind_day[0][ind_step]] = (
                                np.repeat(vector[:, np.newaxis], len(ind_step), axis=1)
                                * ratio[j]
                            )

                        else:
                            input_data.pet[..., ind_day[0][ind_step]] = (
                                np.repeat(
                                    matrix[..., np.newaxis], len(ind_step), axis=2
                                )
                                * ratio[j]
                            )

                    files.pop(ind)

    else:
        for i, date in enumerate(tqdm(date_range, desc="</> Reading pet")):
            date_strf = date.strftime("%Y%m%d%H%M")

            ind = _index_containing_substring(files, date_strf)

            if ind == -1:
                if setup.sparse_storage:
                    input_data.sparse_pet[:, i] = -99.0

                else:
                    input_data.pet[..., i] = -99.0

                warnings.warn(f"Missing pet file for date {date}")

            else:
                matrix = (
                    gdal_read_windowed_raster(
                        filename=files[ind], smash_mesh=mesh, band=1, lacuna=-99.0
                    )
                    * setup.pet_conversion_factor
                )

                if setup.sparse_storage:
                    input_data.sparse_pet[:, i] = sparse_matrix_to_vector(mesh, matrix)

                else:
                    input_data.pet[..., i] = matrix

                files.pop(ind)


def _read_descriptor(setup: SetupDT, mesh: MeshDT, input_data: Input_DataDT):
    for i, name in enumerate(setup.descriptor_name):
        path = glob.glob(
            f"{setup.descriptor_directory}/**/{name}.tif*",
            recursive=True,
        )

        if len(path) == 0:
            warnings.warn(
                f"No descriptor file '{name}.tif' in recursive root directory '{setup.descriptor_directory}'"
            )

        elif len(path) > 1:
            raise ValueError(
                f"There is more than one file containing the name '{name}.tif'"
            )

        else:
            input_data.descriptor[..., i] = gdal_read_windowed_raster(
                filename=path[0], smash_mesh=mesh, band=1, lacuna=-99.0
            )
