enable_testing()
set(SOURCES add_point_test.cpp)
add_executable(add_point_test ${SOURCES})
target_link_libraries(add_point_test PRIVATE
    cuhnsw
    CUDA::cudart
    GTest::gtest_main
)

configure_file(TestConfig.json ${CMAKE_BINARY_DIR}/tests/TestConfig.json COPYONLY)

include(GoogleTest)
gtest_discover_tests(add_point_test)