enable_testing()
add_executable(add_point_test add_point_test.cpp)
target_link_libraries(add_point_test PRIVATE
    cuhnswplus
    CUDA::cudart
    GTest::gtest_main
)

add_executable(index_test index_test.cpp)
target_link_libraries(index_test PRIVATE
    cuhnswplus
    CUDA::cudart
    GTest::gtest_main
)

add_executable(serialization_test serialization_test.cpp)
target_link_libraries(serialization_test PRIVATE
    cuhnswplus
    CUDA::cudart
    GTest::gtest_main
)
configure_file(TestConfig.json ${CMAKE_BINARY_DIR}/tests/cpp/TestConfig.json COPYONLY)

include(GoogleTest)
gtest_discover_tests(add_point_test)