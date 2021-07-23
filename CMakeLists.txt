# Detects whether this is a top-level project
get_directory_property(HAS_PARENT PARENT_DIRECTORY)
if(HAS_PARENT)
    set(BOW_TOPLEVEL_PROJECT OFF)
else()
    set(BOW_TOPLEVEL_PROJECT ON)
endif()

# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.16.0")
if(BOW_TOPLEVEL_PROJECT)
    cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})
else()
    # Don't use cmake_minimum_required here to avoid implicitly overriding parent policies
    if(${CMAKE_VERSION} VERSION_LESS ${REQUIRED_CMAKE_VERSION})
        message(FATAL_ERROR "CMake required version to build Bow is ${REQUIRED_CMAKE_VERSION}")
    endif()
endif()

# Include user-provided default options if available. We do that before the main
# `project()` so that we can define the C/C++ compilers from the option file.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/BowOptions.cmake)
    message(STATUS "Using local options file: ${CMAKE_CURRENT_SOURCE_DIR}/BowOptions.cmake")
    include(${CMAKE_CURRENT_SOURCE_DIR}/BowOptions.cmake)
endif()

################################## OPTIONS ##########################################
option(BOW_STATIC_LIBRARY       "Use bow as static library"                           ON)
option(BOW_STRICT               "Treat compilation warnings as errors"               OFF)
option(BOW_CLANG_FORMAT         "Use clang format tools"                              ON)
option(BOW_UNIT_TESTS           "Compile unit tests"                                  ON)
option(BOW_COMPILE_2D           "[Static library only] compile 2D"                    ON)
option(BOW_COMPILE_3D           "[Static library only] compile 3D"                    ON)
option(BOW_COMPILE_DOUBLE       "[Staic libarary only] compile double precision"      ON)
option(BOW_COMPILE_FLOAT        "[Staic libarary only] compile float precision"       ON)
option(BOW_WITH_SUITESPARSE     "SuiteSparse support for linear solver"               ON)
option(BOW_WITH_AMGCL           "AMGCL support for linear solver"                     ON)
#####################################################################################

project(Bow)

list(PREPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake/recipes/"
    "${PROJECT_SOURCE_DIR}/cmake/modules/"
    "${PROJECT_SOURCE_DIR}/cmake/bow/"
)

if(BOW_TOPLEVEL_PROJECT)
    include(bow_global_flags)
endif()
include(bow_warnings)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

if(BOW_STATIC_LIBRARY)
    add_compile_definitions(BOW_STATIC_LIBRARY)
endif()
if(BOW_COMPILE_2D)
    add_compile_definitions(BOW_COMPILE_2D)
endif()
if(BOW_COMPILE_3D)
    add_compile_definitions(BOW_COMPILE_3D)
endif()
if(BOW_COMPILE_DOUBLE)
    add_compile_definitions(BOW_COMPILE_DOUBLE)
endif()
if(BOW_COMPILE_FLOAT)
    add_compile_definitions(BOW_COMPILE_FLOAT)
endif()

add_subdirectory(include)

add_subdirectory(projects)

if(BOW_CLANG_FORMAT)
    add_subdirectory(format)
endif()

if(BOW_UNIT_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

