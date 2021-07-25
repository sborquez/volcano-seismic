# Utils
import os
import warnings
from pathlib import Path
from tqdm.notebook import tqdm
from IPython.core.display import display, HTML
import wandb as wandb
import datetime
import h5py
import multiprocessing
import collections

# DataScience-CPU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# DataScience-GPU
try:
    import cupy as cp
    import cudf
    import cuml
    __gpu_all__ = ["cp", "cudf", "cuml"]
except:
    __gpu_all__ = []
    warnings.warn("GPU skipped!")

# Distributed DataScience
try:
    import dask_cudf
    import dask
    __dist_all__ = ["dask", "dask_cudf"]
except:
    __dist_all__ = []
    warnings.warn("Dask skipped!")

# Deep Learning
try:
    import tensorflow as tf
    __dl_all__ = ["tf"]
except:
    __dl_all__ = []
    warnings.warn("Tensorflow skipped!")


# Structures    
Worker = collections.namedtuple("Worker", ("queue", "process"))
WorkerInitData = collections.namedtuple("WorkerInitData",
    ("num", "project", "sweep_id", "sweep_run_name", "config", "dataset_artifact_name", "train_index", "test_index", "labels", "num_classes")
)
WorkerDoneData = collections.namedtuple("WorkerDoneData", 
    ("num", "X_test", "y_test", "y_prob", "metrics")
)

def reset_kernel(): 
    os._exit(0)


__all__ = [
    "os", "Path", "tqdm", "display", "HTML",
    "datetime", "plt", "sns", "np", "pd", "sklearn",
    "wandb", "h5py", "multiprocessing", "collections",
    "Worker", "WorkerInitData", "WorkerDoneData",
    "reset_kernel"
]
__all__ += __gpu_all__
__all__ += __dist_all__
__all__ += __dl_all__