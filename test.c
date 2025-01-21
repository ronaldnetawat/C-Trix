#include "matmul.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* Test helpers */
static void fill_random(Matrix* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            matrix_set(mat, i, j, (double)rand() / RAND_MAX);
        }
    }
}

static int matrices_equal(const Matrix* a, const Matrix* b, double epsilon) {
    if (a->rows != b->rows || a->cols != b->cols) return 0;
    
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            MatrixError error;
            double diff = matrix_get(a, i, j, &error) - matrix_get(b, i, j, &error);
            if (fabs(diff) > epsilon) return 0;
        }
    }
    return 1;
}

/* Test cases */
void test_creation_destruction(void) {
    Matrix* mat = matrix_create(3, 4);
    assert(mat != NULL);
    assert(mat->rows == 3);
    assert(mat->cols == 4);
    matrix_free(mat);
    
    mat = matrix_create(0, 4);
    assert(mat == NULL);
}

void test_basic_operations(void) {
    Matrix* mat = matrix_create(2, 2);
    MatrixError error;
    
    assert(matrix_set(mat, 0, 0, 1.0) == MATRIX_SUCCESS);
    assert(matrix_set(mat, 0, 1, 2.0) == MATRIX_SUCCESS);
    assert(matrix_set(mat, 1, 0, 3.0) == MATRIX_SUCCESS);
    assert(matrix_set(mat, 1, 1, 4.0) == MATRIX_SUCCESS);
    
    assert(matrix_get(mat, 0, 0, &error) == 1.0);
    assert(error == MATRIX_SUCCESS);
    
    matrix_free(mat);
}

void test_multiplication(void) {
    Matrix* a = matrix_create(2, 3);
    Matrix* b = matrix_create(3, 2);
    
    double a_values[] = {1, 2, 3, 4, 5, 6};
    double b_values[] = {7, 8, 9, 10, 11, 12};
    
    for (int i = 0; i < 6; i++) {
        matrix_set(a, i/3, i%3, a_values[i]);
        matrix_set(b, i/2, i%2, b_values[i]);
    }
    
    Matrix* c_standard = matrix_multiply(a, b);
    Matrix* c_blocked = matrix_multiply_blocked(a, b);
    Matrix* c_parallel = matrix_multiply_parallel(a, b, 2);
    
    // Verify results match between implementations
    assert(matrices_equal(c_standard, c_blocked, 1e-10));
    assert(matrices_equal(c_standard, c_parallel, 1e-10));
    
    matrix_free(a);
    matrix_free(b);
    matrix_free(c_standard);
    matrix_free(c_blocked);
    matrix_free(c_parallel);
}

void test_performance(void) {
    size_t size = 1500; //matrix size
    Matrix* a = matrix_create(size, size);
    Matrix* b = matrix_create(size, size);
    
    fill_random(a);
    fill_random(b);
    
    BenchmarkResult result = benchmark_multiplication(a, b, 4);
    
    printf("Performance Results (Matrix size: %zux%zu):\n", size, size);
    printf("Standard: %.3f seconds\n", result.standard_time);
    printf("Blocked:  %.3f seconds\n", result.blocked_time);
    printf("Parallel: %.3f seconds\n", result.parallel_time);
    
    matrix_free(a);
    matrix_free(b);
}

int main(void) {
    srand(time(NULL));
    
    printf("Running tests...\n");
    
    test_creation_destruction();
    test_basic_operations();
    test_multiplication();
    test_performance();
    
    printf("All tests passed!\n");
    return 0;
}