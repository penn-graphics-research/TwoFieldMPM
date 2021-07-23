if(TARGET bow::warnings)
    return()
endif()

set(MY_FLAGS
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wcast-align
    -Wformat=2
    -Winit-self
    -Wmissing-include-dirs
    -Woverloaded-virtual
    -Wno-error=redundant-decls
    -fno-math-errno
    -Wno-unused-but-set-parameter
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    -Wno-overloaded-virtual
)

# Flags above don't make sense for MSVC
if(MSVC)
    set(MY_FLAGS)
endif()

include(CheckCXXCompilerFlag)

add_library(bow_warnings INTERFACE)
add_library(bow::warnings ALIAS bow_warnings)

foreach(FLAG IN ITEMS ${MY_FLAGS})
    string(REPLACE "=" "-" FLAG_VAR "${FLAG}")
    if(NOT DEFINED IS_SUPPORTED_${FLAG_VAR})
        check_cxx_compiler_flag("${FLAG}" IS_SUPPORTED_${FLAG_VAR})
    endif()
    if(IS_SUPPORTED_${FLAG_VAR})
        target_compile_options(bow_warnings INTERFACE ${FLAG})
    endif()
endforeach()
