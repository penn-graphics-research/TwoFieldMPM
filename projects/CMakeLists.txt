# libigl
include(libigl)

add_compile_definitions("EXAMPLE_DATA_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/data/\"")

# https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory
# https://stackoverflow.com/questions/13067401/cmake-check-that-a-local-file-exists
macro(listdirs result curdir)
    file(GLOB children RELATIVE ${curdir} ${curdir}/*)
    set(dirlist "")
    foreach(child ${children})
        if(IS_DIRECTORY ${curdir}/${child})
            if(EXISTS "${curdir}/${child}/CMakeLists.txt")
                list(APPEND dirlist ${child})
            endif()
        endif()
    endforeach()
    set(${result} ${dirlist})
endmacro()

listdirs(subdirs ${CMAKE_CURRENT_SOURCE_DIR} )
foreach(subdir ${subdirs})
    message(STATUS "CMake subdirectories: ${subdir}")
    add_subdirectory(${subdir})
endforeach()
