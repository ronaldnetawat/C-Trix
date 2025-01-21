# High-Performance Matrix Multiplication Library with Benchmarking

A C implementation of matrix multiplication optimized for cache efficiency and parallel processing. This library incorporates different approaches to matrix multiplication, from standard implementation to cache-optimized blocked multiplication.

## Features

- Three multiplication methods:
  - Standard Matrix Multiplication (naive approach)
  - Block/Tiled Matrix Multiplication (cache-optimized)
  - Parallel Matrix Multiplication (multi-threaded)
- Benchmarking suite to compare performance
- Cache-aware design using block operations
- Thread-based parallelization
- Comprehensive error handling
- Memory-efficient contiguous array storage

## Performance

Tested on a 1500x1500 matrix:
- Standard: 2.451 seconds
- Parallel: 1.873 seconds
- Blocked:  0.924 seconds

The blocked implementation shows significant performance improvements due to better cache utilization.

## Requirements

- GCC or Clang compiler
- POSIX threads support
- Math library

## Installation

```bash
gcc -O3 -pthread matrix.c matrix_test.c -o matrix_test -lm
```

## Usage

Run the test suite:
```bash
./matrix_test
```

Include in your project:
```c
#include "matrix.h"

// Create matrices
Matrix* a = matrix_create(rows, cols);
Matrix* b = matrix_create(cols, other_cols);

// Multiply using different methods
Matrix* result_standard = matrix_multiply(a, b);
Matrix* result_blocked = matrix_multiply_blocked(a, b);
Matrix* result_parallel = matrix_multiply_parallel(a, b, num_threads);

// Free memory
matrix_free(a);
matrix_free(b);
matrix_free(result_standard);
matrix_free(result_blocked);
matrix_free(result_parallel);
```

## File Structure

- `matrix.h` - Header file with function declarations and data structures
- `matrix.c` - Implementation of matrix operations
- `matrix_test.c` - Test suite and benchmarking code

## Implementation Details

- Uses contiguous memory layout for better cache performance
- Implements block multiplication for cache efficiency
- Employs loop unrolling for improved instruction pipelining
- Includes thread-based parallelization
- Features comprehensive error handling

## License

[Your chosen license]

## Contributing

Feel free to open issues or submit pull requests.

## Author

[Your name]

## Acknowledgments

This implementation draws inspiration from cache-optimization techniques in high-performance computing.
