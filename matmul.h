#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

//configurable param for optimization
#define BLOCK_SIZE 32   //cache blocking size: should be tuned based on your CPU's L1 cache
#define UNROLL_SIZE 4  //loop unrolling factor

// error codes for error handling
typedef enum {
    MATRIX_SUCCESS,
    MATRIX_INVALID_DIMENSIONS,
    MATRIX_MEMORY_ERROR,
    MATRIX_NULL_POINTER,
    MATRIX_INDEX_OUT_OF_BOUNDS
} MatrixError;

// matrix structure using row-major layout for better cache performance
typedef struct {
    size_t rows;
    size_t cols;
    double* data;     // Flat array for better memory locality
    MatrixError error;   //last error that occurred
} Matrix;

/* Core matrix operations */
Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix* mat);
Matrix* matrix_multiply(const Matrix* a, const Matrix* b);
Matrix* matrix_multiply_blocked(const Matrix* a, const Matrix* b);
Matrix* matrix_multiply_parallel(const Matrix* a, const Matrix* b, int num_threads);

// utility funcs
MatrixError matrix_set(Matrix* mat, size_t row, size_t col, double value);
double matrix_get(const Matrix* mat, size_t row, size_t col, MatrixError* error);
Matrix* matrix_copy(const Matrix* src);
void matrix_print(const Matrix* mat);

//performance measurement utilities
typedef struct {
    double standard_time;
    double blocked_time;
    double parallel_time;
    size_t matrix_size;
    int num_threads;
} BenchmarkResult;

BenchmarkResult benchmark_multiplication(const Matrix* a, const Matrix* b, int num_threads);

#endif // MATRIX_H