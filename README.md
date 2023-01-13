Python Package for SnowWhite
============================

This is the Python front end for the [SPIRAL project's](http://www.spiral.net) SnowWhite system, which compiles high-level specifications of numerical computations into hardware-specific optimized code.  It supports a variety of CPU's as well as Nvidia and AMD GPU's on systems running Linux, Windows, or MacOS.

SnowWhite originated under the [DARPA PAPPA](https://www.darpa.mil/program/performant-automation-of-parallel-program-assembly) (Performant Automation of Parallel Program Assembly) program.  The program focused on ways reduce the complexity of building software that takes advantage of the massive parallelism of advanced high-preformance computing systems.  Further work continued as part of [FFTX](http://www.spiral.net/software/fftx.html) under the [Exascale Computing Project](https://www.exascaleproject.org/).

SnowWhite, implemented as a Python package, uses the SPIRAL code generation system to translate NumPy-based specifications to generated code, then compiles that code into a loadable library.

For details of the structure and API of the package, use the Python `help()` function to view the 
internal documentation of the installed package.  For example:

```python console
>>> import snowwhite
>>> help(snowwhite)
```

See the `examples` directory for a quick introduction.

*DISTRIBUTION STATEMENT A.  Approved for public release.  Distribution is unlimited.*

## Prerequisites

- **Python3** (3.7 or higher)
	- **NumPy**
	- **CuPy** (optional, but needed if targeting GPU's)
- **SPIRAL** (available on GitHub)
	- **spiral-software** https://www.github.com/spiral-software/spiral-software
	- **spiral-package-fftx** https://www.github.com/spiral-software/spiral-package-fftx
	- **spiral-package-simt** https://www.github.com/spiral-software/spiral-package-simt
- **CMake** (3.14 or higher)
- **C Compiler**

With SPIRAL installed you will have CMake, a compatible C compiler, and Python3 for SnowWhite.  SPIRAL builds on Linux/Unix with gcc and make, on Windows it builds with Visual Studio.  For macOS SPIRAL requires version 10.14 (Mojave) or later of macOS, with a compatible version of Xcode and
and Xcode Command Line Tools. 


## Installing and Configuring SPIRAL

Clone **spiral-software** to a location on your computer.  For example:
```shell
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME* and you must set an environment variable
**SPIRAL_HOME** to point to this location later.

To install the two spiral packages do the following:
```shell
cd ~/work/spiral-software/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
```
**NOTE:** The spiral packages must be installed under the directory
**$SPIRAL_HOME/namespaces/packages** and must be placed in directories with the
prefix "spiral-package-" removed. 

Follow the [build instructions](https://github.com/spiral-software/spiral-software/blob/master/README.md) for **spiral-software**.


## Installing and Configuring SnowWhite

Clone **python-package-snowwhite** to a location on your computer renaming it to **snowwhite**.  For example:
```shell
cd ~/work/python-packages
git clone https://github.com/spiral-software/python-package-snowwhite snowwhite
```

Add the directory that contains the **snowwhite** clone to the environment variable **PYTHONPATH**.  (In the above example that would be ```~/work/python-packages```.)  This allows Python to locate the **snowwhite** module.

By default, SnowWhite puts generated files into a temporary directory under the current working directory, then deletes that temporary directory after a successful build.  If there is an error during the build, the temporary directory will remain.  There are two environment variables that can modify this default behavior:

+ **SW_WORKDIR** specifies the path to the parent directory of the temporary build directories.  If that specified directory does not exist, SnowWhite uses the current directory.

+ **SW_KEEPTEMP** if defined (any value) tells SnowWhite to preserve temporary build directories.


## Exernal Libraries

**Snowwhite** can access libraries built by [**FFTX**](https://github.com/spiral-software/fftx), which have metadata that describes their contents.  SnowWhite looks in its ```.libs``` directory for any libraries containing compatible metadata.  It also looks for libraries in directories specified by the **SW_LIBRARY_PATH** environment variable, with the list of directories having the same format as used for the **PATH** variable.


## Try an Example

Open a terminal window in the ```examples``` directory and run this example:

```shell
~/work/python-packages/snowwhite/examples$ python run-mddft.py 32
```
The first time you run it, you will see output from the CMake/SPIRAL/C build before the comparison results,
similar to this:

```shell
Generating CUDA
Compiling and linking
Diff between Python/C transforms = 1.206500728403121e-12
```

After that it will run much faster using the generated library, which is placed in the ```snowwhite/.libs``` directory.

Some of the examples require additional arguments, and some options you can change.  Read through the examples for better understanding.  If you want to see the generated source files, set the **SW_KEEPTEMP** environment variable and look in the temporary directories.






