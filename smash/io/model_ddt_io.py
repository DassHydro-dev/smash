from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smash.core.model import Model

from smash.core._constant import STRUCTURE_PARAMETERS, STRUCTURE_STATES

from smash.io._error import ReadHDF5MethodError

import os
import errno
import warnings
import h5py
import numpy as np

__all__ = ["save_model_ddt", "read_model_ddt"]


def _default_save_data(structure: str):
    return {
        "setup": ["dt", "end_time", "start_time", "structure"],
        "mesh": ["active_cell", "area", "code", "dx", "flwdir"],
        "input_data": ["mean_prcp", "mean_pet", "qobs"],
        "parameters": STRUCTURE_PARAMETERS[
            structure
        ],  # only calibrated Model param will be stored
        "states": STRUCTURE_STATES[
            structure
        ],  # only initial Model states will be stored
        "output": [
            {
                "fstates": STRUCTURE_STATES[structure]
            },  # only final Model states will be stored
            "qsim",
        ],
    }


def _parse_selected_derived_type_to_hdf5(
    derived_type, list_attr, hdf5_ins, attr_suffix=""
):
    # TODO: clean function for attr_suffix

    for attr in list_attr:
        if isinstance(attr, str):
            try:
                value = getattr(derived_type, attr)

                attr += attr_suffix

                if isinstance(value, np.ndarray):
                    if value.dtype == "object" or value.dtype.char == "U":
                        value = value.astype("S")

                    hdf5_ins.create_dataset(
                        attr,
                        shape=value.shape,
                        dtype=value.dtype,
                        data=value,
                        compression="gzip",
                        chunks=True,
                    )

                else:
                    hdf5_ins.attrs[attr] = value

            except:
                pass

        elif isinstance(attr, dict):
            for derived_type_key, list_attr_imd in attr.items():
                try:
                    derived_type_imd = getattr(derived_type, derived_type_key)

                    _parse_selected_derived_type_to_hdf5(
                        derived_type_imd, list_attr_imd, hdf5_ins
                    )

                except:
                    pass


def save_model_ddt(model: Model, path: str, sub_data=None, sub_only=False):
    """
    Save some derived data types of the Model object to HDF5 file.

    This method is considerably lighter than `smash.save_model` method that saves the entire Model object.
    However, it is not capable of reconstructing the Model object from the saved data file.

    By default, the following data are stored into the `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ file:

    - ``dt``, ``end_time``, ``start_time``, ``structure`` from `Model.setup`
    - ``active_cell``, ``area``, ``code``, ``dx``, ``flwdir`` from `Model.mesh`
    - ``mean_prcp``, ``mean_pet``, ``qobs`` from `Model.input_data`
    - ``qsim`` from `Model.output`
    - The final Model states (depending upon the Model structure) from state derived type of `Model.output`
    - The initial Model states (depending upon the Model structure) from `Model.states`
    - The Model parameters (depending upon the Model structure) from `Model.parameters`

    Subsidiary data can be added by filling in ``sub_data``.

    Parameters
    ----------
    model : Model
        The Model object to save derived data types as a HDF5 file.

    path : str
        The file path. If the path not end with ``.hdf5``, the extension is automatically added to the file path.

    sub_data : dict or None, default None
        Dictionary which indicates the subsidiary data to store into the HDF5 file.

        .. note::
            If not given, no subsidiary data is saved

    sub_only : bool, default False
        Allow to only store subsidiary data.

    See Also
    --------
    read_model_ddt: Read derived data types of the Model object from HDF5 file.
    Model: Primary data structure of the hydrological model `smash`.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> model
    Structure: 'gr-a'
    Spatio-Temporal dimension: (x: 28, y: 28, time: 1440)
    Last update: Initialization

    Save spatially distributed precipitation in addition to default derived data types of Model

    >>> smash.save_model_ddt(model, "model_ddt.hdf5", sub_data={"prcp": model.input_data.prcp})

    """

    if not path.endswith(".hdf5"):
        path = path + ".hdf5"

    with h5py.File(path, "w") as f:
        if not sub_only:
            save_data = _default_save_data(model.setup.structure)

            for derived_type_key, list_attr in save_data.items():
                derived_type = getattr(model, derived_type_key)

                if derived_type_key == "states":
                    _parse_selected_derived_type_to_hdf5(
                        derived_type, list_attr, f, attr_suffix="_0"
                    )

                else:
                    _parse_selected_derived_type_to_hdf5(derived_type, list_attr, f)

        if sub_data is not None:
            for attr, value in sub_data.items():
                if (attr in f) or (attr in f.attrs):
                    warnings.warn(f"Ignore updating existing key ({attr})")

                    continue

                if isinstance(value, np.ndarray):
                    if value.dtype == "object" or value.dtype.char == "U":
                        value = value.astype("S")

                    try:
                        f.create_dataset(
                            attr,
                            shape=value.shape,
                            dtype=value.dtype,
                            data=value,
                            compression="gzip",
                            chunks=True,
                        )

                    except:
                        warnings.warn(f"Can not store to HDF5: {attr}")

                else:
                    try:
                        f.attrs[attr] = value

                    except:
                        warnings.warn(f"Can not store to HDF5: {attr}")

        f.attrs["_save_func"] = "save_model_ddt"


def read_model_ddt(path: str) -> dict:
    """
    Read derived data types of the Model object from HDF5 file.

    Parameters
    ----------
    path : str
        The file path.

    Returns
    -------
    data : dict
        A dictionary with derived data types loaded from HDF5 file.

    Raises
    ------
    FileNotFoundError:
        If file not found.
    ReadHDF5MethodError:
        If file not created with `save_model_ddt`.

    See Also
    --------
    save_model_ddt: Save some derived data types of the Model object to HDF5 file.

    Examples
    --------
    >>> setup, mesh = smash.load_dataset("cance")
    >>> model = smash.Model(setup, mesh)
    >>> smash.save_model_ddt(model, "model_ddt.hdf5")

    Read the derived data types from HDF5 file

    >>> data = smash.read_model_ddt("model_ddt.hdf5")

    Then, to see the dataset keys

    >>> data.keys()
    dict_keys(['active_cell', 'area', 'cft', 'code', 'cp', 'exc', 'flwdir',
    'hft', 'hft_0', 'hlr', 'hlr_0', 'hp', 'hp_0', 'lr', 'mean_pet', 'mean_prcp',
    'qobs', 'qsim', 'dt', 'dx', 'end_time', 'start_time', 'structure'])

    And finally, to access to derived data

    >>> data["mean_prcp"]
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)

    """

    if os.path.isfile(path):
        with h5py.File(path) as f:
            if f.attrs.get("_save_func") == "save_model_ddt":
                keys = list(f.keys())

                values = [
                    f[key][:].astype("U") if f[key][:].dtype.char == "S" else f[key][:]
                    for key in keys
                ]

                attr_keys = list(f.attrs.keys())

                attr_keys.remove("_save_func")

                attr_values = [f.attrs[key] for key in attr_keys]

                return dict(zip(keys + attr_keys, values + attr_values))

            else:
                raise ReadHDF5MethodError(
                    f"Unable to read '{path}' with 'read_model_ddt' method. The file may not have been created with 'save_model_ddt' method."
                )

    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
