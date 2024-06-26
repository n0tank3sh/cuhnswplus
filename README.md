# CUHNSWPLUS

It is an improved fork of [cuhnsw](https://github.com/js1010/cuhnsw), featuring enhanced performance and new functionalities.

## Improvements
- **Multithreading**: Parallel loading of graph vectors for faster processing.
- **Performance**: Reuses allocated `device_vector` to reduce overhead.
- **Disk Caching**: Efficient handling of large datasets using disk caching.
- **API Support**: Added support for `AddPoint` and `AddPoints` functions.

## Installation
### Python: 
```sh
pip install .
```
### C++:
```sh
cmake -S . -B build
cd build
make install -j`nproc`
```

## Usage:
Copy the example config from ``examples/examples_config.json``
### Python
Refer to the `examples` folder for detailed usage examples.
### CPP
Refer to the `tests` folder for detailed usage examples.

## Contributing

We welcome contributions! Please feel free to open issues, submit pull requests, or provide feedback.