set(SOURCE_FILES
        src/cuhnsw_base.cu
        src/cuhnsw_build.cu
        src/cuhnsw_search.cu
        src/log.cc
        src/index.cpp
        )
set(HEADER_FILES
        include/cuda_base_kernels.cuh
        include/cuda_build_kernels.cuh
        include/cuda_dist_kernels.cuh
        include/cuda_heap_kernels.cuh
        include/cuda_search_kernels.cuh
        include/cuda_utils_kerenels.cuh
        include/cuhnsw.hpp
        include/level_graph.hpp
        include/log.hpp
        include/stop_watch.hpp
        include/types.hpp
        )
add_library(cuhnsw ${SOURCE_FILES})
set_target_properties(cuhnsw PROPERTIES CUDA_ARCHITECTURES "all-major")
target_include_directories(cuhnsw PUBLIC include/)
target_link_libraries(cuhnsw json11 spdlog CCCL::CCCL CUDA::cudart CUDA::cufft)