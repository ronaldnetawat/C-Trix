# Matrix Multiplication Library with Benchmarking from Scratch in C

A C implementation of matrix multiplication optimized for cache efficiency and parallel processing. This library incorporates different approaches to matrix multiplication, from standard implementation to cache-optimized blocked multiplication.

## Features

- Three multiplication methods:
  - Standard Matrix Multiplication (naive approach)
  - Block/Tiled Matrix Multiplication (cache-optimized)
  - Parallel Matrix Multiplication (multi-threaded)
- Benchmarking to compare performance
- Cache-aware design using block operations
- Thread-based parallelization
- Error handling
- Memory-efficient flat array storage

## Performance

Tested on a 1500x1500 matrix:
- Standard: 4.165 seconds
- Parallel: 1.388 seconds
- Blocked:  1.250 seconds

The blocked implementation shows significant performance improvements due to better cache utilization.

## Requirements

- GCC or Clang compiler
- POSIX threads support
- Math library

## Installation

```bash
gcc -O3 -pthread matrix.c matrix_test.c -o matrix_test -lm
```


## File Structure

- `matmul.h` - Header file with function declarations and data structures
- `matmul.c` - Implementation of matrix operations
- `test.c` - Test suite and benchmarking code


## Contributing

Feel free to open issues or submit pull requests.
