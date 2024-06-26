set(SOURCE_FILES
        src/cuhnswplus_base.cu
        src/cuhnswplus_build.cu
        src/cuhnswplus_search.cu
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
        include/cuhnswplus.hpp
        include/level_graph.hpp
        include/log.hpp
        include/stop_watch.hpp
        include/types.hpp
        )
add_library(cuhnswplus ${SOURCE_FILES})
set_target_properties(cuhnswplus PROPERTIES CUDA_ARCHITECTURES "all-major")
target_include_directories(cuhnswplus PUBLIC include/ ../3rd/thread-pool/include)
target_link_libraries(cuhnswplus json11 spdlog CCCL::CCCL CUDA::cudart CUDA::cufft)