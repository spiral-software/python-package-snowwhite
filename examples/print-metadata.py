'''Print meta data in shared library files in snowwhite .libs directory
'''

import os
import snowwhite
from snowwhite.metadata import *
from pprint import *

moduleDir = os.path.dirname(os.path.realpath(snowwhite.metadata.__file__))
libsDir = os.path.join(moduleDir, SW_LIBSDIR)
md = metadataInDir(libsDir)
pprint(md)