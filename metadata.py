
from snowwhite import *

import json
import glob
import os

def metadataInFile(filename):
    """extract metadata from binary file"""
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
    """Assemble metadata (if any) from shared library files in directory"""
    metalist = []
    filepat = os.path.join(path, '*.dll')
    files = glob.glob(filepat)
    for filename in files:
        metaobj = metadataInFile(filename)
        if metaobj != None:
            metalist.append({'filename':filename, 'metadata':metaobj})
    return metalist


def writeMetadataSourceFile(metadata, varname, path):
    """Write metadata JSON as compileable C string"""
    try:
        metadata_file = open(path, 'w')
    except:
        print('Error: Could not open ' + path + ' for writing')
        return
    metastr = json.dumps(metadata, sort_keys=True, indent=4)
    metastr = metastr.replace('"', '\\"') + '\\'
    metastr = metastr.replace('\n', '\\\n')
    print('char *' + varname + ' = "' + SW_METADATA_START + '\\', file = metadata_file)  
    print(metastr, file = metadata_file) 
    print(SW_METADATA_END + '";', file = metadata_file)  
    metadata_file.close()

