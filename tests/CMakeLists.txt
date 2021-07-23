include(catch2)
FetchContent_GetProperties(catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

file(GLOB TEST_SOURCES "*.h" "*.cpp")
add_executable(bow_tests ${TEST_SOURCES})
target_link_libraries(bow_tests PUBLIC
    bow::bow
    bow::warnings
    Catch2::Catch2
)

catch_discover_tests(bow_tests)
