##
## Copyright (c) 2018-2020, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")
set ( FILEROOT "hockney_stub" CACHE STRING "when problem is defined will get correct/full name" )
set ( HASCUDA OFF CACHE BOOL "when true build for CUDA")
set ( HASMPI OFF CACHE BOOL "when true build for MPI")

if ( NOT DEFINED PY_LIBS_DIR )
	set ( PY_LIBS_DIR ${CMAKE_SOURCE_DIR} )
endif ()


if (${HASCUDA})
	project (SPIRAL_CUDA
         VERSION 1.0.1
                DESCRIPTION "SPIRAL CUDA code generation"
                LANGUAGES C CUDA)
	if (${HASMPI})
		set ( SOURCES ${FILEROOT}.cu mpimain.cu)
	else ()
		set ( SOURCES ${FILEROOT}.cu ${FILEROOT}_host.cu )
	endif ()
else ()
	project (SPIRAL_CPU
         VERSION 1.0.1
                DESCRIPTION "SPIRAL C code generation"
                LANGUAGES C CXX)
	set ( SOURCES ${FILEROOT}.c )
endif ()


##  Setup flags if needed

if (WIN32)
    set (CUDA_COMPILE_FLAGS -DWIN64 -rdc=false)
else ()
    set (CUDA_COMPILE_FLAGS -m64 -rdc=true)
endif ()

if (${HASMPI})
	set (CUDA_COMPILE_FLAGS ${CUDA_COMPILE_FLAGS} -DFUNCNAME=${FILEROOT})
endif ()


if (WIN32)
    set ( GPU_COMPILE_DEFNS )
else ()
    set ( GPU_COMPILE_DEFNS -dc )
endif ()



if ( WIN32 )
    set ( LIB_PREF lib )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON )
else ()
    set ( LIB_PREF )
endif ()

set ( PROJECT ${LIB_PREF}${FILEROOT} )

add_library ( ${PROJECT} SHARED ${SOURCES} )
##  target_compile_options     (${PROJECT}  PRIVATE )		## any flags needed?

if (${HASCUDA})
	set_property(TARGET ${PROJECT} PROPERTY CUDA_ARCHITECTURES "60;70;72;75;80")
	set ( CMAKE_CUDA_ARCHITECTURES 70 )
	target_compile_options     (${PROJECT} PRIVATE ${CUDA_COMPILE_FLAGS} ${GPU_COMPILE_DEFNS})
	set_property        (TARGET ${PROJECT} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON )
endif()

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( INC_DIR $ENV{SPIRAL_HOME}/profiler/targets )
	set ( INC_DIR2 $ENV{SPIRAL_HOME}/profiler/targets/include )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
	message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( INC_DIR ${SPIRAL_HOME}/profiler/targets )
endif ()
    
message ( STATUS "Profiler target directory = ${INC_DIR}" )
include_directories (${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${INC_DIR} ${INC_DIR2} )

install (TARGETS ${PROJECT} DESTINATION ${PY_LIBS_DIR})

