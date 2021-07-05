# Utils
import os
import warnings
from pathlib import Path
from tqdm.notebook import tqdm
from IPython.core.display import display, HTML
#import wandb as wandb
import datetime

# DataScience-CPU
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    __dist_all__ = ["dask_cudf"]
except:
    __dist_all__ = []
    warnings.warn("Dask skipped!")

# Deep Learning
try:
    import tensorflow as tf
    __dl_all__ = ["tf"]
except:
    __dl_all__ = []
    warnings.warn("Dask skipped!")


def reset_kernel(): 
    os._exit(00)

__all__ = [
    "os", "Path", "tqdm", "display", "HTML",
    "datetime", "plt", "sns", "np", "pd", 
    "reset_kernel" 
]
__all__ += __gpu_all__
__all__ += __dist_all__
__all__ += __dl_all__