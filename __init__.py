
from .swsolver import *

import numpy as _numpy

try:
    import cupy as _cupy
except ModuleNotFoundError:
    _cupy = None



def get_array_module(*args):
    if _cupy != None:
        return _cupy.get_array_module(*args)
    else:
        return _numpy


def has_ROCm():
    if _cupy != None:
        return (_cupy._environment.get_rocm_path() != None)
    else:
        return False

