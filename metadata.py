
from snowwhite import *

import json
import glob
import os
import sys

def metadataInFile(filename):
    """extract metadata from binary file."""
    bstr = bytes(SW_METADATA_START, 'utf-8')
    estr = bytes(SW_METADATA_END, 'utf-8')
    with open(filename, 'rb') as f:
        buff = f.read()
        if bstr in buff:
            b = buff.find(bstr) + len(bstr)
            e = buff.find(estr)
            metabytes = buff[b:e]
            metaobj = json.loads(metabytes)
            return metaobj
        else:
            return None


def metadataInDir(path):
    """Assemble metadata from shared library files in directory."""
    metalist = []
    filepat = os.path.join(path, '*' + SW_SHLIB_EXT)
    files = glob.glob(filepat)
    for filename in files:
        metaobj = metadataInFile(filename)
        if metaobj != None:
            metalist.append({SW_KEY_FILENAME:filename, SW_KEY_METADATA:metaobj})
    return metalist


def writeMetadataSourceFile(metadata, varname, path, spaces=0):
    """Write metadata JSON as compileable C string."""
    try:
        metadata_file = open(path, 'w')
    except:
        print('Error: Could not open ' + path + ' for writing')
        return
    metastr = json.dumps(metadata, sort_keys=True, indent=spaces)
    metastr = metastr.replace('"', '\\"') + '\\'
    metastr = metastr.replace('\n', '\\\n')
    print('char *' + varname + ' = "' + SW_METADATA_START + '\\', file = metadata_file)  
    print(metastr, file = metadata_file) 
    print(SW_METADATA_END + '";', file = metadata_file)  
    metadata_file.close()
    
    
def metadataMatches(metadata, metavals):
    if len(metavals) < 1:
        return False
    for k,v in metavals.items():
        if not k in metadata:
            return False
        if v != metadata[k]:
            return False
    return True
    
    
def findFunctionsWithMetadata(metavals, libdir=None):
    """Search for matching metadata in libraries."""
    if not type(metavals) is dict:
        return(None, None)
        
    transformType = metavals.get(SW_KEY_TRANSFORMTYPE)
    if transformType == None:
        return(None, None)    
        
    if libdir == None:
        moduleDir = os.path.dirname(os.path.realpath(__file__))
        libdir = os.path.join(moduleDir, SW_LIBSDIR)
        
    dirlist = [libdir]
    
    libpath = os.getenv(SW_LIBRARY_PATH)
    if libpath != None:
        sep = ';' if sys.platform == 'win32' else ':'
        paths = libpath.split(sep)
        dirlist = dirlist + paths
        
    for libdir in dirlist:    
        mdlist = metadataInDir(libdir)
        for filedict in mdlist:
            filemd = filedict.get(SW_KEY_METADATA, {})
            if transformType not in filemd.get(SW_KEY_TRANSFORMTYPES, []):
                continue
            for xform in filemd.get(SW_KEY_TRANSFORMS, []):
                if metadataMatches(xform, metavals):
                    return(filedict.get(SW_KEY_FILENAME), xform.get(SW_KEY_NAMES, {}))
            
    return (None, None)


