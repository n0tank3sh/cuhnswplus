find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(CuHNSWBind bindings.cc)
target_link_libraries(CuHNSWBind PUBLIC cuhnswplus)