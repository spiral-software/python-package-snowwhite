
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

SPIRAL_RET_OK   = 0
SPIRAL_RET_ERR  = 1

if sys.platform == 'win32':
    SPIRAL_EXE = 'spiral.bat'
else:
    SPIRAL_EXE = 'spiral'


def spiralBuildInfo():
    # -B option signals Spiral to print build info and exit early in startup
    # use BuildInfo() and quit commands for older Spiral version w/o -B option
    fallthroughstr = b'BuildInfo();\nquit;\n'
    try:
        res = subprocess.run([SPIRAL_EXE, '-B'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=fallthroughstr)
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


def callSpiralWithFile(filename):
    try:
        with open(filename, 'r') as f:
            runResult = subprocess.run(SPIRAL_EXE, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if runResult.returncode == 0:
                return SPIRAL_RET_OK
            else:
                print(runResult.stderr.decode(), file=sys.stderr)
                return SPIRAL_RET_ERR
    except OSError as ex:
        print(ex.strerror, file=sys.stderr)
    except:
        pass
    return SPIRAL_RET_ERR

