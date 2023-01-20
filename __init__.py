
from .spiral import *

import sys
import numpy as _numpy

try:
    import cupy as _cupy
except ModuleNotFoundError:
    _cupy = None

__version__ = '1.0.1a1'
    
# internal names

SW_LIBSDIR  = '.libs'

# environment varibles

SW_KEEPTEMP     = 'SW_KEEPTEMP'
SW_LIBRARY_PATH = 'SW_LIBRARY_PATH'
SW_WORKDIR      = 'SW_WORKDIR'

# options

SW_OPT_COLMAJOR         = 'colmajor'
SW_OPT_KEEPTEMP         = 'keeptemp'
SW_OPT_METADATA         = 'metadata'
SW_OPT_MPI              = 'mpi'
SW_OPT_PLATFORM         = 'platform'
SW_OPT_PRINTRULETREE    = 'printruletree'
SW_OPT_REALCTYPE        = 'realctype'

# transform direction, 'k'

SW_FORWARD  = -1
SW_INVERSE  = 1

# platforms

SW_CPU  = 'CPU'
SW_CUDA = 'CUDA'
SW_HIP  = 'HIP'

# metadata

SW_METADATA_START   = '!!START_METADATA!!'
SW_METADATA_END     = '!!END_METADATA!!'
SW_METAFILE_EXT     = '_meta.c'
SW_METAVAR_EXT      = '_metadata'

SW_STR_DOUBLE       = 'Double'
SW_STR_SINGLE       = 'Single'

SW_STR_FORWARD      = 'Forward'
SW_STR_INVERSE      = 'Inverse'

SW_TRANSFORM_DFT        = 'DFT'
SW_TRANSFORM_MDDFT      = 'MDDFT'
SW_TRANSFORM_MDRCONV    = 'MDRCONV'
SW_TRANSFORM_MDRFSCONV  = 'MDRFSCONV'
SW_TRANSFORM_MDPRDFT    = 'MDPRDFT'
SW_TRANSFORM_UNKNOWN    = 'UNKNOWN'

SW_KEY_DESTROY          = 'Destroy'
SW_KEY_DIMENSIONS       = 'Dimensions'
SW_KEY_DIRECTION        = 'Direction'
SW_KEY_EXEC             = 'Exec'
SW_KEY_FILENAME         = 'Filename'
SW_KEY_FUNCTIONS        = 'Functions'
SW_KEY_INIT             = 'Init'
SW_KEY_METADATA         = 'Metadata'
SW_KEY_NAMES            = 'Names'
SW_KEY_PLATFORM         = 'Platform'
SW_KEY_PRECISION        = 'Precision'
SW_KEY_SPIRALBUILDINFO  = 'SpiralBuildInfo'
SW_KEY_TRANSFORMS       = 'Transforms'
SW_KEY_TRANSFORMTYPE    = 'TransformType'
SW_KEY_TRANSFORMTYPES   = 'TransformTypes'

if sys.platform == 'win32':
    SW_SHLIB_EXT = '.dll'
elif sys.platform == 'darwin':
    SW_SHLIB_EXT = '.dylib'
else:
    SW_SHLIB_EXT = '.so'


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

