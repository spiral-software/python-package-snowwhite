'''Print meta data in shared library files in .libs directory

script expects to be in examples directory in order to find .libs
'''

import os
from snowwhite.metadata import *
from pprint import *

examplesDir = os.path.dirname(os.path.realpath(__file__))
libsDir = os.path.join(examplesDir, '../.libs')
md = metadataInDir(libsDir)
pprint(md)