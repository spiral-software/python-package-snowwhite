
import sys
import subprocess


SPIRAL_KEY_CMAKEVERSION     =  'CMakeVersion'
SPIRAL_KEY_COMPILER         =  'Compiler'
SPIRAL_KEY_CONFIGURATION    =  'Configuration'
SPIRAL_KEY_DATETIMEUTC      =  'DateTimeUTC'
SPIRAL_KEY_GITBRANCH        =  'GitBranch'
SPIRAL_KEY_GITHASH          =  'GitHash'
SPIRAL_KEY_GITREMOTE        =  'GitRemote'
SPIRAL_KEY_SYSTEM           =  'System'
SPIRAL_KEY_VERSION          =  'Version'


def spiralBuildInfo():
    if sys.platform == 'win32':
        spiralexe = 'spiral.bat'
    else:
        spiralexe = 'spiral'

    # -B option signals Spiral to print build info and exit early in startup
    # use BuildInfo() and quit commands for older Spiral version w/o -B option
    fallthroughstr = b'BuildInfo();\nquit;\n'
    try:
        res = subprocess.run([spiralexe, '-B'], capture_output=True, input=fallthroughstr)
    except:
        return dict()
    bdl = res.stdout.split(b'\n')
    bdd = dict()
    for item in bdl:
        kv = item.split(b':',1)
        if len(kv) == 2:
            key = kv[0].strip().decode()
            val = kv[1].strip().decode()
            bdd[key] = val
    return bdd

