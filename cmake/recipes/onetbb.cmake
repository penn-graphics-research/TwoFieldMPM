if(TARGET TBB::tbb)
    return()
endif()

message(STATUS "Third-party: creating targets 'TBB::tbb'")

include(FetchContent)
FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG b3fb83948504388cc6f85286b1d2063334df008b
)

option(TBB_TEST "Enable testing" OFF)
option(TBB_EXAMPLES "Enable examples" OFF)
option(TBB_STRICT "Treat compiler warnings as errors" ON)
set(BUILD_SHARED_LIBS OFF)
unset(TBB_DIR CACHE)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tbb)
FetchContent_MakeAvailable(tbb)

if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB::tbb is still not defined!")
endif()

foreach(name IN ITEMS tbb tbbmalloc tbbmalloc_proxy)
    if(TARGET ${name})
        set_target_properties(${name} PROPERTIES FOLDER "third_party//tbb")
    endif()
endforeach()
