#pragma once
#include "data_generation.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

/*
 * throws an exception if an errorCode is meet
 */
void checkCudaErrorCode(cudaError const errorCode, std::string const &functionName)
{
    if (errorCode != cudaSuccess)
    {
        throw std::runtime_error(functionName + ": " + cudaGetErrorString(errorCode));
    }
}

/*
 * allocates an array of the given size on the device
 * returns a pointer to the array
 * similar to `new T[size]`
 *
 * use `cudaMallocManaged` to be able to touche memory on host and allocate more than available on GPU
 */
template<typename T>
T *cudaNew(size_t size)
{
    T *pointer;
    cudaError const errorCode = cudaMallocManaged((void **)&pointer, sizeof(T) * size);
    checkCudaErrorCode(errorCode, "cudaMalloc");
    return pointer;
}

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 */
template<typename T>
class DeviceArrayBatch
{
  public:
    // lets you access the device pointer directly
    T **rawPointer;
    size_t nb_arrays;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    DeviceArrayBatch(size_t const array_sizes, size_t const nb_arrays_arg,
                     bool const should_initialize_data = false)
        : nb_arrays(nb_arrays_arg)
    {
        // gets device for prefetch
        int device = -1;
        cudaGetDevice(&device);
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = cudaNew<T *>(nb_arrays);
        for (unsigned int i = 0; i < nb_arrays; i++)
        {
            rawPointer[i] = cudaNew<T>(array_sizes);
            if (should_initialize_data) fillArray(rawPointer[i], array_sizes, rng);
            cudaMemPrefetchAsync(rawPointer[i], array_sizes * sizeof(T), device);
        }
        cudaMemPrefetchAsync(rawPointer, nb_arrays * sizeof(T *), device);
    }

    // releases the memory
    ~DeviceArrayBatch()
    {
        // frees the batch elements
        for (unsigned int i = 0; i < nb_arrays; i++)
        {
            cudaError const errorCode = cudaFree(rawPointer[i]);
            checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch[])");
        }
        // free the array of batch elements
        cudaError const errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch)");
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 * make sure that the batch onctains only a given number of distinct elements
 */
template<typename T>
class DeviceArrayBatch_withRepetition
{
  public:
    // lets you access the device pointer directly
    T **rawPointer;
    size_t array_sizes;
    size_t nb_arrays;
    size_t nb_arrays_distinct;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    // contains only `nb_arrays_distinct` distinct elemnts
    DeviceArrayBatch_withRepetition(size_t const array_sizes_arg, size_t const nb_arrays_arg,
                                    size_t const nb_arrays_distinct_arg = 5,
                                    bool const should_initialize_data   = false)
        : array_sizes(array_sizes_arg), nb_arrays(nb_arrays_arg), nb_arrays_distinct(nb_arrays_distinct_arg)
    {
        // gets device for prefetch
        int device = -1;
        cudaGetDevice(&device);
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = cudaNew<T *>(nb_arrays);
        for (unsigned int i = 0; i < nb_arrays_distinct; i++)
        {
            rawPointer[i] = cudaNew<T>(array_sizes);
            if (should_initialize_data) fillArray(rawPointer[i], array_sizes, rng);
            cudaMemPrefetchAsync(rawPointer[i], array_sizes * sizeof(T), device);
        }
        // allocates blocks of identical batch elements
        for (unsigned int i = nb_arrays_distinct; i < nb_arrays; i++)
        {
            int const ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            rawPointer[i]       = rawPointer[ptr_index];
        }
        cudaMemPrefetchAsync(rawPointer, nb_arrays * sizeof(T *), device);
    }

    // deep copy constructor
    DeviceArrayBatch_withRepetition(DeviceArrayBatch_withRepetition const &arraybatch)
        : DeviceArrayBatch_withRepetition(arraybatch.array_sizes, arraybatch.nb_arrays, arraybatch.nb_arrays_distinct)
    {
        for (unsigned int i = 0; i < nb_arrays_distinct; i++)
        {
            cudaMemcpy(rawPointer[i], arraybatch.rawPointer[i], sizeof(T) * array_sizes, cudaMemcpyDefault);
        }
    }

    // releases the memory
    ~DeviceArrayBatch_withRepetition()
    {
        // frees the batch elements
        for (unsigned int i = 0; i < nb_arrays_distinct; i++)
        {
            cudaError const errorCode = cudaFree(rawPointer[i]);
            checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch[])");
        }
        // free the array of batch elements
        cudaError const errorCode = cudaFree(rawPointer);
        checkCudaErrorCode(errorCode, "cudaFree (~DeviceArrayBatch)");
    }

    // computes the maximum relative distance between two arraybatch
    T distance(DeviceArrayBatch_withRepetition const &arraybatch)
    {
        T const epsilon = 1e-15;
        T max_dist      = 0.;

        for (unsigned int i = 0; i < nb_arrays_distinct; i++)
        {
            T *v1       = rawPointer[i];
            T const *v2 = arraybatch.rawPointer[i];
            for (unsigned int j = 0; j < array_sizes; j++)
            {
                T const dist = std::abs(v1[i] - v2[i]) / (std::abs(v1[i]) + epsilon);
                if (dist > max_dist) max_dist = dist;
            }
        }

        return max_dist;
    }
};