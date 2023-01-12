"""
Print meta data in shared library files in snowwhite .libs directory 
and in directories specified in SW_LIBRARY_PATH
"""

import os
import snowwhite
from snowwhite.metadata import *
from pprint import *

moduleDir = os.path.dirname(os.path.realpath(snowwhite.metadata.__file__))
libsDir = os.path.join(moduleDir, SW_LIBSDIR)

dirlist = [libsDir]

libpath = os.getenv(SW_LIBRARY_PATH)
if libpath != None:
    sep = ';' if sys.platform == 'win32' else ':'
    paths = libpath.split(sep)
    dirlist = dirlist + paths

md = []
for libdir in dirlist:    
    md = md + metadataInDir(libdir)

pprint(md)