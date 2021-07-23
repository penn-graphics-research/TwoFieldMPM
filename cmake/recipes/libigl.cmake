if(TARGET igl::core)
    return()
endif()

message(STATUS "Third-party: creating target 'igl::core'")

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 0bb27beb88ae1aebe9653754c64d3f281f7e4a33
)
FetchContent_GetProperties(libigl)
if(libigl_POPULATED)
    return()
endif()
FetchContent_Populate(libigl)

include(eigen)

list(APPEND CMAKE_MODULE_PATH ${libigl_SOURCE_DIR}/cmake)
include(${libigl_SOURCE_DIR}/cmake/libigl.cmake ${libigl_BINARY_DIR})

# Install rules
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME libigl)
set_target_properties(igl PROPERTIES EXPORT_NAME core)
install(DIRECTORY ${libigl_SOURCE_DIR}/include/igl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS igl igl_common EXPORT Libigl_Targets)
install(EXPORT Libigl_Targets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/igl NAMESPACE igl::)
