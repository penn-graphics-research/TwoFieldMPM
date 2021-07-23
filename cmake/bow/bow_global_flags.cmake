set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT WIN32)
    if(BOW_STRICT)
        add_compile_options(-Werror=all)
    endif()

    add_compile_options(-march=native)
endif()
