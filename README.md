# Kronmult

This library implements `kronmult(matrix_list, vector)`, which is the matrix product of the kronecker product of several matrices and a given vector, for both GPU (using CUDA) and CPU (using OpenMP).

It is fine-tuned for the needs of [ASGarD](https://github.com/project-asgard/asgard).
In particular, we expect our inputs to be *col-major* square matrices.

# Theory

We implement a variant of the backward version of algorithm 993 ([Algorithm 993: Efficient Computation with Kronecker Products](https://dl.acm.org/doi/abs/10.1145/3291041)), chosen to perform well on col-major matrices.

We highly recommend reading [ON KRONECKER PRODUCTS, TENSOR PRODUCTS AND MATRIX DIFFERENTIAL CALCULUS by Stephen Pollock](https://www.le.ac.uk/economics/research/RePEc/lec/leecon/dp14-02.pdf) to get more familiar with the algebra and reshaping tricks techniques used in the implementation.

## Usage

We implemented a basic and a batched version of the algorithm:

```cpp
#include <kronmult/kronmult_openmp.hpp>

void kronmult_openmp::kronmult(const int matrix_number, const int matrix_size, T const * const matrix_list[], const int matrix_stride,
                               T input[], const int size_input,
                               T workspace[], T transpose_workspace[])

void kronmult_openmp::kronmult_batched(const int matrix_number, const int matrix_size, T const * const matrix_list_batched[], const int matrix_stride,
                                       T* input_batched[],
                                       T* output_batched[], T* workspace_batched[],
                                       const int nb_batch)
```

They both compute `output += kron(matrix_list) * input`.

- `matrix_list` is an array containing pointers to `matrix_number` square matrices of size `matrix_size` by `matrix_size` and stride `matrix_stride`
- `input` is a `size_input` = `matrix_size`^`matrix_number` elements vector
- `output` is a `size_input` elements vector, where the output will be stored
- `workspace` is a `size_input` elements vector, to be used as workspace
- `transpose_workspace` is a `matrix_size`^2 elements vector, to be used as workspace

In the batched version all inputs are replaced by arrays with one input for each of the `nb_batch` batch elements except for `matrix_list_batched` which is an array of `nb_batch`*`matrix_number` pointers to square matrices.

**WARNINGS**:

- `input` and `workspace` will be used as temporary workspaces and thus modified
- the matrices should be stored in col-major order

## Compilation

To compile the code for CPU, run:

```
mkdir build && cd build
cmake ../
make
```

To compile the code for an Nvidia GPU, pass the `-DUSE_GPU=1` flag to CMake:

```
mkdir build && cd build
cmake ../ -DUSE_GPU=1
make
```

## Branch specificity: batched_gemm_mkl
To compile and run with mkl, the line 
`set(BLA_VENDOR Intel10_64lp)` has been added to CMakeLists.txt in tests.
This is specific to Pinocchio the INRIA Corse INTEL avx512 machine.