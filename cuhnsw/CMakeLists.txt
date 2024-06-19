find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cuhnsw_bind bindings.cc)
target_link_libraries(cuhnsw_bind PUBLIC cuhnsw)