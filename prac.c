/*
 * matrix.h - Main header file containing all public interfaces
 */
#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/* Configurable parameters for optimization */
#define BLOCK_SIZE 32    // Cache blocking size - tune based on your CPU's L1 cache
#define UNROLL_SIZE 4    // Loop unrolling factor

/* Error codes for robust error handling */
typedef enum {
    MATRIX_SUCCESS,
    MATRIX_INVALID_DIMENSIONS,
    MATRIX_MEMORY_ERROR,
    MATRIX_NULL_POINTER,
    MATRIX_INDEX_OUT_OF_BOUNDS
} MatrixError;

/* Matrix structure using row-major layout for better cache performance */
typedef struct {
    size_t rows;
    size_t cols;
    double* data;        // Flat array for better memory locality
    MatrixError error;   // Last error that occurred
} Matrix;

/* Core matrix operations */
Matrix* matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix* mat);
Matrix* matrix_multiply(const Matrix* a, const Matrix* b);
Matrix* matrix_multiply_blocked(const Matrix* a, const Matrix* b);
Matrix* matrix_multiply_parallel(const Matrix* a, const Matrix* b, int num_threads);

/* Utility functions */
MatrixError matrix_set(Matrix* mat, size_t row, size_t col, double value);
double matrix_get(const Matrix* mat, size_t row, size_t col, MatrixError* error);
Matrix* matrix_copy(const Matrix* src);
void matrix_print(const Matrix* mat);

/* Performance measurement utilities */
typedef struct {
    double standard_time;
    double blocked_time;
    double parallel_time;
    size_t matrix_size;
    int num_threads;
} BenchmarkResult;

BenchmarkResult benchmark_multiplication(const Matrix* a, const Matrix* b, int num_threads);

#endif // MATRIX_H

/*
 * matrix.c - Core implementation file
 */
#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

/* Helper macro for index calculation in flat array */
#define IDX(i, j, cols) ((i) * (cols) + (j))

/* Internal helper functions */
static inline int is_valid_matrix(const Matrix* mat) {
    return mat != NULL && mat->data != NULL;
}

Matrix* matrix_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }

    Matrix* mat = malloc(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }

    /* Allocate contiguous memory for better cache performance */
    mat->data = calloc(rows * cols, sizeof(double));
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->error = MATRIX_SUCCESS;

    return mat;
}

void matrix_free(Matrix* mat) {
    if (mat) {
        free(mat->data);
        free(mat);
    }
}

/* Standard matrix multiplication - baseline implementation */
Matrix* matrix_multiply(const Matrix* a, const Matrix* b) {
    if (!is_valid_matrix(a) || !is_valid_matrix(b) || a->cols != b->rows) {
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) {
        return NULL;
    }

    /* Basic multiplication with pointer arithmetic */
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            double sum = 0.0;
            double* arow = &a->data[IDX(i, 0, a->cols)];
            double* bcol = &b->data[IDX(0, j, b->cols)];
            
            for (size_t k = 0; k < a->cols; k++) {
                sum += arow[k] * bcol[k * b->cols];
            }
            
            result->data[IDX(i, j, result->cols)] = sum;
        }
    }

    return result;
}

/* Cache-efficient blocked multiplication with loop unrolling */
Matrix* matrix_multiply_blocked(const Matrix* a, const Matrix* b) {
    if (!is_valid_matrix(a) || !is_valid_matrix(b) || a->cols != b->rows) {
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) {
        return NULL;
    }

    /* Zero initialize result matrix */
    memset(result->data, 0, sizeof(double) * result->rows * result->cols);

    /* Blocked multiplication */
    for (size_t i = 0; i < a->rows; i += BLOCK_SIZE) {
        size_t imax = (i + BLOCK_SIZE < a->rows) ? i + BLOCK_SIZE : a->rows;
        
        for (size_t j = 0; j < b->cols; j += BLOCK_SIZE) {
            size_t jmax = (j + BLOCK_SIZE < b->cols) ? j + BLOCK_SIZE : b->cols;
            
            for (size_t k = 0; k < a->cols; k += BLOCK_SIZE) {
                size_t kmax = (k + BLOCK_SIZE < a->cols) ? k + BLOCK_SIZE : a->cols;
                
                /* Process current block */
                for (size_t ii = i; ii < imax; ii++) {
                    for (size_t jj = j; jj < jmax; jj++) {
                        double sum = result->data[IDX(ii, jj, result->cols)];
                        size_t kk;
                        
                        /* Unrolled inner loop */
                        for (kk = k; kk + UNROLL_SIZE <= kmax; kk += UNROLL_SIZE) {
                            sum += a->data[IDX(ii, kk, a->cols)] * b->data[IDX(kk, jj, b->cols)] +
                                  a->data[IDX(ii, kk+1, a->cols)] * b->data[IDX(kk+1, jj, b->cols)] +
                                  a->data[IDX(ii, kk+2, a->cols)] * b->data[IDX(kk+2, jj, b->cols)] +
                                  a->data[IDX(ii, kk+3, a->cols)] * b->data[IDX(kk+3, jj, b->cols)];
                        }
                        
                        /* Handle remaining elements */
                        for (; kk < kmax; kk++) {
                            sum += a->data[IDX(ii, kk, a->cols)] * b->data[IDX(kk, jj, b->cols)];
                        }
                        
                        result->data[IDX(ii, jj, result->cols)] = sum;
                    }
                }
            }
        }
    }

    return result;
}

/* Structure for parallel multiplication thread data */
typedef struct {
    const Matrix* a;
    const Matrix* b;
    Matrix* result;
    size_t start_row;
    size_t end_row;
} ThreadData;

/* Thread function for parallel multiplication */
void* multiply_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    for (size_t i = data->start_row; i < data->end_row; i += BLOCK_SIZE) {
        size_t imax = (i + BLOCK_SIZE < data->end_row) ? i + BLOCK_SIZE : data->end_row;
        
        for (size_t j = 0; j < data->b->cols; j += BLOCK_SIZE) {
            size_t jmax = (j + BLOCK_SIZE < data->b->cols) ? j + BLOCK_SIZE : data->b->cols;
            
            for (size_t k = 0; k < data->a->cols; k += BLOCK_SIZE) {
                size_t kmax = (k + BLOCK_SIZE < data->a->cols) ? k + BLOCK_SIZE : data->a->cols;
                
                /* Process block using same optimization as blocked multiplication */
                for (size_t ii = i; ii < imax; ii++) {
                    for (size_t jj = j; jj < jmax; jj++) {
                        double sum = data->result->data[IDX(ii, jj, data->result->cols)];
                        size_t kk;
                        
                        for (kk = k; kk + UNROLL_SIZE <= kmax; kk += UNROLL_SIZE) {
                            sum += data->a->data[IDX(ii, kk, data->a->cols)] * data->b->data[IDX(kk, jj, data->b->cols)] +
                                  data->a->data[IDX(ii, kk+1, data->a->cols)] * data->b->data[IDX(kk+1, jj, data->b->cols)] +
                                  data->a->data[IDX(ii, kk+2, data->a->cols)] * data->b->data[IDX(kk+2, jj, data->b->cols)] +
                                  data->a->data[IDX(ii, kk+3, data->a->cols)] * data->b->data[IDX(kk+3, jj, data->b->cols)];
                        }
                        
                        for (; kk < kmax; kk++) {
                            sum += data->a->data[IDX(ii, kk, data->a->cols)] * data->b->data[IDX(kk, jj, data->b->cols)];
                        }
                        
                        data->result->data[IDX(ii, jj, data->result->cols)] = sum;
                    }
                }
            }
        }
    }

    return NULL;
}

/* Parallel matrix multiplication */
Matrix* matrix_multiply_parallel(const Matrix* a, const Matrix* b, int num_threads) {
    if (!is_valid_matrix(a) || !is_valid_matrix(b) || a->cols != b->rows || num_threads <= 0) {
        return NULL;
    }

    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) {
        return NULL;
    }

    /* Zero initialize result matrix */
    memset(result->data, 0, sizeof(double) * result->rows * result->cols);

    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    ThreadData* thread_data = malloc(num_threads * sizeof(ThreadData));

    if (!threads || !thread_data) {
        matrix_free(result);
        free(threads);
        free(thread_data);
        return NULL;
    }

    /* Divide work among threads */
    size_t rows_per_thread = a->rows / num_threads;
    size_t extra_rows = a->rows % num_threads;
    size_t current_row = 0;

    for (int i = 0; i < num_threads; i++) {
        thread_data[i].a = a;
        thread_data[i].b = b;
        thread_data[i].result = result;
        thread_data[i].start_row = current_row;
        
        size_t rows_this_thread = rows_per_thread + (i < extra_rows ? 1 : 0);
        current_row += rows_this_thread;
        thread_data[i].end_row = current_row;

        pthread_create(&threads[i], NULL, multiply_thread, &thread_data[i]);
    }

    /* Wait for all threads to complete */
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    free(threads);
    free(thread_data);

    return result;
}

/* Utility functions implementation */
MatrixError matrix_set(Matrix* mat, size_t row, size_t col, double value) {
    if (!is_valid_matrix(mat)) {
        return MATRIX_NULL_POINTER;
    }
    
    if (row >= mat->rows || col >= mat->cols) {
        return MATRIX_INDEX_OUT_OF_BOUNDS;
    }

    mat->data[IDX(row, col, mat->cols)] = value;
    return MATRIX_SUCCESS;
}

double matrix_get(const Matrix* mat, size_t row, size_t col, MatrixError* error) {
    if (!is_valid_matrix(mat)) {
        if (error) *error = MATRIX_NULL_POINTER;
        return 0.0;
    }
    
    if (row >= mat->rows || col >= mat->cols) {
        if (error) *error = MATRIX_INDEX_OUT_OF_BOUNDS;
        return 0.0;
    }

    if (error) *error = MATRIX_SUCCESS;
    return mat->data[IDX(row, col, mat->cols)];
}

Matrix* matrix_copy(const Matrix* src) {
    if (!is_valid_matrix(src)) {
        return NULL;
    }

    Matrix* copy = matrix_create(src->rows, src->cols);
    if (!copy) {
        return NULL;
    }

    memcpy(copy->data, src->data, sizeof(double) * src->rows * src->cols);
    return copy;
}

void matrix_print(const Matrix* mat) {
    if (!is_valid_matrix(mat)) {
        printf("Invalid matrix\n");
        return;
    }

    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%8.2f ", mat->data[IDX(i, j, mat->cols)]);
        }
        printf("\n");
    }
}

/* Benchmarking implementation */
BenchmarkResult benchmark_multiplication(const Matrix* a, const Matrix* b, int num_threads) {
    BenchmarkResult result = {0};
    result.matrix_size = a->rows;  // Assuming square matrices
    result.num_threads = num_threads;

    clock_t start, end;
    Matrix *c;

    /* Time standard multiplication */
    start = clock();
    c = matrix_multiply(a, b);
    end = clock();
    result.standard_time = (double)(end - start) / CLOCKS_PER_SEC;
    matrix_free(c);

    /* Time blocked multiplication */
    start = clock();
    c = matrix_multiply_blocked(a, b);
    end = clock();
    result.blocked_time = (double)(end - start) / CLOCKS_PER_SEC;
    matrix_free(c);

    /* Time parallel multiplication */
    start = clock();
    c = matrix_multiply_parallel(a, b, num_threads);
    end = clock();
    result.parallel_time = (double)(end - start) / CLOCKS