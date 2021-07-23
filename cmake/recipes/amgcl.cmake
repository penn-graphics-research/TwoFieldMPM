if(TARGET amgcl::amgcl)
    return()
endif()

message(STATUS "Third-party: creating target 'amgcl::amgcl'")

include(FetchContent)
FetchContent_Declare(
    amgcl
    GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
    GIT_TAG 461a66ce6d197a3816218bf94ffd114a367c1ef1
)

function(amgcl_import_target)
    macro(ignore_package NAME VERSION_NUM)
        include(CMakePackageConfigHelpers)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}Config.cmake "")
        write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/${NAME}/${NAME}ConfigVersion.cmake
            VERSION ${VERSION_NUM}
            COMPATIBILITY AnyNewerVersion
        )
        set(${NAME}_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "")
        set(${NAME}_ROOT ${CMAKE_CURRENT_BINARY_DIR}/${NAME} CACHE PATH "")
    endmacro()

    include(boost)

    ignore_package(Boost 1.71.0)
    set(Boost_INCLUDE_DIRS "")
    set(Boost_LIBRARIES "")

    # Prefer Config mode before Module mode to prevent lib from loading its own FindXXX.cmake
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)

    # Ready to include third-party lib
    FetchContent_MakeAvailable(amgcl)

    target_link_libraries(amgcl INTERFACE Boost::boost)
endfunction()

amgcl_import_target()
