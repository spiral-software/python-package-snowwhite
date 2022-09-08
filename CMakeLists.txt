##
## Copyright (c) 2018-2022, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

set (CMAKE_BUILD_TYPE Release  CACHE STRING "Debug, Release, RelWithDebInfo, MinSizeRel")
set ( FILEROOT "undefined" CACHE STRING "unique transform root name for source and library files" )
set ( HASCUDA OFF CACHE BOOL "when true build for CUDA")
set ( HASHIP OFF CACHE BOOL "when true build for HIP")
set ( HASMPI OFF CACHE BOOL "when true build for MPI")
set ( HAS_METADATA OFF CACHE BOOL "when true include metadata file in build")

if ( NOT DEFINED PY_LIBS_DIR )
    set ( PY_LIBS_DIR ${CMAKE_SOURCE_DIR} )
endif ()


if ( ${HASCUDA} )
    ##  Build for CUDA is defined
    project ( SPIRAL_CUDA
        VERSION 1.0.1
        DESCRIPTION "SPIRAL CUDA code generation"
        LANGUAGES C CUDA )
    if ( ${HASMPI} )
	set ( SOURCES ${FILEROOT}.cu mpimain.cu )
    else ()
	set ( SOURCES ${FILEROOT}.cu )
    endif ()

elseif ( ${HASHIP} )
    ##  Build for HIP is defined
    set ( CMAKE_CXX_COMPILER "hipcc" )
    project ( SPIRAL_HIP
	VERSION 1.0.1
	DESCRIPTION "SPIRAL HIP code generation"
	LANGUAGES C CXX )
    set ( SOURCES ${FILEROOT}.cpp )

    ##  Setup what we need to build for HIP/ROCm
    list ( APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm )
    find_package ( hip REQUIRED )
    if ( ${hip_FOUND} )
	##  HIP/ROCm support found
	message ( STATUS "Found HIP: Version = ${hip_VERSION}" )
    else ()
	message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build" )
    endif ()

    ##  Adjust include and library directories
    ##  Need to add $ROCM_PATH for includes and libraries
    if ( DEFINED ENV{ROCM_PATH} )
	message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
	include_directories ( $ENV{ROCM_PATH}/hipfft/include $ENV{ROCM_PATH}/include )
    endif ()
    list ( APPEND LIBS_FOR_HIP hipfft rocfft )

else ()
    ##  No GPU selected, build for CPU
    project ( SPIRAL_CPU
        VERSION 1.0.1
        DESCRIPTION "SPIRAL C code generation"
        LANGUAGES C CXX )
    set ( SOURCES ${FILEROOT}.c )
endif ()

if ( ${HAS_METADATA} )
	list ( APPEND SOURCES ${FILEROOT}_meta.c )
endif()


##  Setup flags if needed

if (WIN32)
    set (CUDA_COMPILE_FLAGS -DWIN64 -rdc=false)
    set ( GPU_COMPILE_DEFNS )
    set ( LIB_PREF lib )
    set ( CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON )
else ()
    ##  set (CUDA_COMPILE_FLAGS -m64 -rdc=true)
    ##  set ( GPU_COMPILE_DEFNS -dc )
    set ( LIB_PREF )
endif ()

if (${HASMPI})
    list ( APPEND CUDA_COMPILE_FLAGS -DFUNCNAME=${FILEROOT} )
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
    set ( INC_DIR $ENV{SPIRAL_HOME}/profiler/targets
		  $ENV{SPIRAL_HOME}/profiler/targets/include )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
	message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( INC_DIR ${SPIRAL_HOME}/profiler/targets  ${SPIRAL_HOME}/profiler/targets/include )
endif ()
    
message ( STATUS "Profiler target directory = ${INC_DIR}" )
target_include_directories ( ${PROJECT} PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} ${INC_DIR} )

install (TARGETS ${PROJECT} DESTINATION ${PY_LIBS_DIR})

