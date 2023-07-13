from __future__ import annotations

from smash.core.simulation._optimize import (
    _optimize_sbs,
    _optimize_nelder_mead,
    _optimize_lbfgsb,
)

import numpy as np


### STRUCTURE ###
#################

STRUCTURE_PARAMETERS = {
    "gr-a": ["cp", "cft", "exc", "lr"],
    "gr-b": ["ci", "cp", "cft", "exc", "lr"],
    "gr-c": ["ci", "cp", "cft", "cst", "exc", "lr"],
    "gr-d": ["cp", "cft", "lr"],
    "vic-a": ["b", "cusl1", "cusl2", "clsl", "ks", "ds", "dsm", "ws", "lr"],
}

STRUCTURE_STATES = {
    "gr-a": ["hp", "hft", "hlr"],
    "gr-b": ["hi", "hp", "hft", "hlr"],
    "gr-c": ["hi", "hp", "hft", "hst", "hlr"],
    "gr-d": ["hp", "hft", "hlr"],
    "vic-a": ["husl1", "husl2", "hlsl"],
}

STRUCTURE_ADJUST_CI = {
    "gr-a": False,
    "gr-b": True,
    "gr-c": True,
    "gr-d": False,
    "vic-a": False,
}

STRUCTURE_NAME = list(STRUCTURE_PARAMETERS.keys())


### READ INPUT DATA ###
#######################

INPUT_DATA_FORMAT = ["tif", "nc"]

RATIO_PET_HOURLY = np.array(
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
)


### SIGNATURES ###
##################

CSIGN = ["Crc", "Crchf", "Crclf", "Crch2r", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]

ESIGN = ["Eff", "Ebf", "Erc", "Erchf", "Erclf", "Erch2r", "Elt", "Epf"]

SIGNS = CSIGN + ESIGN


### PRCP INDICES ###
####################

PRCP_INDICES = ["std", "d1", "d2", "vg"]


### OPTIMIZE ###
################

ALGORITHM = ["sbs", "l-bfgs-b", "nelder-mead"]

JOBS_FUN = [
    "nse",
    "kge",
    "kge2",
    "se",
    "rmse",
    "logarithmic",
]

CSIGN_OPTIM = ["Crc", "Cfp2", "Cfp10", "Cfp50", "Cfp90"]

ESIGN_OPTIM = ["Erc", "Elt", "Epf"]

MAPPING = ["uniform", "distributed", "hyper-linear", "hyper-polynomial"]

GAUGE_ALIAS = ["all", "downstream"]

WGAUGE_ALIAS = ["mean", "median", "area", "minv_area"]

# % TODO: Add "distance_correlation" once clearly verified
JREG_FUN = [
    "prior",
    "smoothing",
    "hard_smoothing",
]

AUTO_WJREG = ["fast", "lcurve"]

OPTIM_FUNC = {
    "sbs": _optimize_sbs,
    "l-bfgs-b": _optimize_lbfgsb,
    "nelder-mead": _optimize_nelder_mead,
}  # % TODO add nsga


### NET ###
###########

WB_INITIALIZER = [
    "uniform",
    "glorot_uniform",
    "he_uniform",
    "normal",
    "glorot_normal",
    "he_normal",
    "zeros",
]

NET_OPTIMIZER = ["sgd", "adam", "adagrad", "rmsprop"]

LAYER_NAME = ["dense", "activation", "scale", "dropout"]


### GENERATE SAMPLES ###
########################

SAMPLE_GENERATORS = ["uniform", "normal", "gaussian"]

PROBLEM_KEYS = ["num_vars", "names", "bounds"]
