#pragma once
#include <cstdlib>

/*
 * encapsulates new in a function
 */
template<typename T>
T *new_function(size_t const size)
{
    return new T[size];
}
// types used to encapsulate malloc and free in either gpu or cpu execution
template<typename T>
using MallocFunction = T *(size_t);
using FreeFunction   = void(void *);

/*
 * converts row and col indices into a single index for a matrix store in col-major
 * `stride` is usually the number of rows of the matrix
 */
constexpr int colmaj(int const row, int const col, int const stride)
{
    return row + col * stride;
}

/*
 * naive implementation of a matrix vector product
 * we suppose that the matrix is square, stored in col-major and of stride one
 * the result is *added* to output
 */
template<typename T>
void matrix_vector_product(T const matrix[], int const size, int const stride, T const vector[], T output[])
{
    for (int row = 0; row < size; row++)
    {
        for (int col = 0; col < size; col++)
        {
            output[row] += matrix[colmaj(row, col, stride)] * vector[col];
        }
    }
}

/*
 * does the kronecker product of two square col-major matrices
 * the output is a square col-major matrix of stride its number of columns
 */
template<typename T>
void kronecker_product(T const matrix1[], int const size1, int const stride1, T const matrix2[],
                       int const size2, int const stride2, T output[])
{
    int const stride_out = size1 * size2;
    for (int row1 = 0; row1 < size1; row1++)
    {
        for (int row2 = 0; row2 < size2; row2++)
        {
            for (int col1 = 0; col1 < size1; col1++)
            {
                for (int col2 = 0; col2 < size2; col2++)
                {
                    int const row_out = row1 * size2 + row2;
                    int const col_out = col1 * size2 + col2;
                    output[colmaj(row_out, col_out, stride_out)] = matrix1[colmaj(row1, col1, stride1)] * matrix2[colmaj(row2, col2, stride2)];
                }
            }
        }
    }
}

/*
 * naive implementation of kronmult
 * you can pass a GPU malloc and free to insure it works without segfaulting on gpu data
 */
template<typename T>
void kronmult_naive(int const matrix_count, int const matrix_size, T *matrix_list[], int const matrix_stride,
                    T const input[], T output[], MallocFunction<T> malloc_f = new_function,
                    FreeFunction free_f = free)
{
    // computes the kronnecker product
    T *kronmat      = matrix_list[0];
    int size_kron   = matrix_size;
    int stride_kron = matrix_stride;
    for (int m = 1; m < matrix_count; m++)
    {
        // allocates new kronmat
        T *kronmat_new;
        kronmat_new = malloc_f(size_kron * matrix_size);
        // does kronecker product
        T const *matrix = matrix_list[m];
        kronecker_product(kronmat, size_kron, stride_kron, matrix, matrix_size, matrix_stride, kronmat_new);
        // replace old kronmat
        // do not delete input matrix
        if (m > 1) free_f(kronmat);
        kronmat     = kronmat_new;
        size_kron   = size_kron * matrix_size;
        stride_kron = size_kron;
    }
    // does the matrix vector product and adds the result to output
    matrix_vector_product(kronmat, size_kron, stride_kron, input, output);
    // frees the memory
    // do not delete input matrix
    if (matrix_count > 1) free_f(kronmat);
}

/*
 * naive implementation of batched kronmult
 * you can pass a GPU malloc and free to insure it works without segfaulting on gpu data
 */
template<typename T>
void kronmult_batched_naive(int const matrix_count, int const matrix_size, T* matrix_list_batched[],
                            int const matrix_stride, T* input_batched[], T* output_batched[],
                            T* /*workspace_batched*/[], int const nb_batch,
                            MallocFunction<T> malloc_f = new_function, FreeFunction free_f = free)
{
    for (int i = 0; i < nb_batch; i++)
    {
        T **matrix_list = &matrix_list_batched[i * matrix_count];
        T *input        = input_batched[i];
        T *output       = output_batched[i];
        kronmult_naive(matrix_count, matrix_size, matrix_list, matrix_stride, input, output, malloc_f, free_f);
    }
}
