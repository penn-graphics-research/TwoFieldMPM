# Common dependencies
include(eigen)
include(tbb)

# Spgrid
add_library(spgrid
        SPGrid/Core/SPGrid_Geometry.cpp
        SPGrid/Core/SPGrid_Utilities.cpp)
add_library(spgrid::spgrid ALIAS spgrid)
target_include_directories(spgrid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spgrid
    PRIVATE
        bow::warnings
    PUBLIC
        Eigen3::Eigen
        TBB::tbb
)
target_compile_features(spgrid PUBLIC cxx_std_11)

if(BOW_STATIC_LIBRARY)
    message(STATUS "Compiling Bow as a static library")
    file(GLOB_RECURSE SOURCES Bow/*.cpp)
    add_library(bow_lib ${SOURCES})
    add_library(bow::bow ALIAS bow_lib)
    target_include_directories(bow_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(bow_lib PRIVATE bow::warnings)
    set(BOW_SCOPE PUBLIC)
else()
    add_library(bow_lib INTERFACE)
    add_library(bow::bow ALIAS bow_lib)
    target_include_directories(bow_lib SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    set(BOW_SCOPE INTERFACE)
endif()

target_link_libraries(bow_lib ${BOW_SCOPE}
    Eigen3::Eigen
    TBB::tbb
    spgrid
)
target_compile_features(bow_lib ${BOW_SCOPE} cxx_std_17)

if(BOW_WITH_AMGCL)
    include(amgcl)
    target_link_libraries(bow_lib ${BOW_SCOPE} amgcl::amgcl)
    target_compile_definitions(bow_lib ${BOW_SCOPE} BOW_AMGCL)
endif()

if(BOW_WITH_SUITESPARSE)
    set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITESPARSE_INC})
    set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITESPARSE_LIB})
    find_package(SuiteSparse REQUIRED)
    target_link_libraries(bow_lib ${BOW_SCOPE} ${SUITESPARSE_LIBRARIES})
    target_include_directories(bow_lib ${BOW_SCOPE} ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(bow_lib ${BOW_SCOPE} BOW_SUITESPARSE)
endif()
