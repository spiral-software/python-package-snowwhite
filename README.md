Python Module for SnowWhite
===========================

This is the prototype Python front end for the SPIRAL project's SnowWhite system, which compiles high-level specifications of numerical computations into hardware-specific optimized code.  It supports a variety of CPU's as well as Nvidia and AMD GPU's on systems running Linux, Windows, or MacOS.

SnowWhite was developed under the DARPA PAPPA (Performant Automation of Parallel Program Assembly) program.  The program focused on ways reduce the complexity of building software that takes advantage of the massive parallelism of advanced high-preformance computing systems.

SmowWhite, implemented as a Python module, uses the SPIRAL code generation system to translate NumPy-based specifications to generated code, then compiles that code into a loadable library.

See the ```examples``` directory to learn more.

*DISTRIBUTION STATEMENT A.  Approved for public release.  Distribution is unlimited.*

## Prerequisites

Besides a current version of Python3 and NumPy, SnowWhite needs the following:

- **SPIRAL** (available on GitHub)
	- **spiral-software** https://www.github.com/spiral-software/spiral-software
	- **spiral-package-fftx** https://www.github.com/spiral-software/spiral-package-fftx
	- **spiral-package-simt** https://www.github.com/spiral-software/spiral-package-simt
- **CMake**
- **C Compiler**

With SPIRAL installed you will have CMake, a compatible C compiler, and Python3 for SnowWhite.  SPIRAL builds on Linux/Unix with **gcc** and **make**, on Windows it builds with **Visual Studio**.  For macOS SPIRAL requires version 10.14 (Mojave) or later of macOS, with a compatible version of **Xcode** and
and **Xcode Command Line Tools**. 



## Installing and Configuring SPIRAL

Clone **spiral-software** to a location on you computer.  For example:
```
cd ~/work
git clone https://www.github.com/spiral-software/spiral-software
```
This location is known as *SPIRAL HOME* and you must set an environment variable
**SPIRAL_HOME** to point to this location later.

To install the two spiral packages do the following:
```
cd ~/work/spiral-software/namespaces/packages
git clone https://www.github.com/spiral-software/spiral-package-fftx fftx
git clone https://www.github.com/spiral-software/spiral-package-simt simt
```
**NOTE:** The spiral packages must be installed under directory
**$SPIRAL_HOME/namespaces/packages** and must be placed in folders with the
prefix *spiral-package* removed. 

Follow the [build instructions](https://github.com/spiral-software/spiral-software/blob/master/README.md) for **spiral-software**.


## Installing and Configuring SnowWhite

Clone **python-package-snowwhite** to a location on you computer renaming it to **snowwhite**.  For example:
```
cd ~/work
git clone https://github.com/spiral-software/python-package-snowwhite snowwhite
```

Add the directory that contains the **snowwhite** clone to the environment variable **PYTHONPATH**.  (In the above example that would be ```~/work```.)  This allows Python to locate the **snowwhite** module.

## Try an Example

Copy one of the example Python scripts from the ```examples``` directory to a scratch directory and run it like this:

```
D:\Temp>python run-hockney8.py
```

The first time you run it, you will see output from the CMake/SPIRAL/C build, but after that it will run much faster using the generated library, which is placed in the ```snowwhite/.libs``` directory.

Some of the examples require additional arguments, and some options you can change.  Read through the examples for better understanding, and examine the generated intermediate files in you scratch directory.








