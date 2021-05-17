#pragma once
#include <vector>
#include <cstring>
#include "data_generation.h"

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 */
template<typename T>
class ArrayBatch
{
  public:
    // lets you access the data pointer directly
    T** rawPointer;
    size_t nb_arrays;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    ArrayBatch(const size_t array_sizes, const size_t nb_arrays_arg, const bool should_initialize_data=false): nb_arrays(nb_arrays_arg)
    {
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = new T*[nb_arrays];
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            rawPointer[i] = new T[array_sizes];
            if(should_initialize_data) fillArray(rawPointer[i], array_sizes, rng);
        }
    }

    // releases the memory
    ~ArrayBatch()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays; i++)
        {
            delete[] rawPointer[i];
        }
        // free the array of batch elements
        delete[] rawPointer;
    }
};

/*
 * wraps a batch of arrays on device with a proper constructor and destructor
 * make sure that the batch onctains only a given number of distinct elements
 */
template<typename T>
class ArrayBatch_withRepetition
{
  public:
    // lets you access the device pointer directly
    T** rawPointer;
    size_t array_sizes;
    size_t nb_arrays;
    size_t nb_arrays_distinct;

    // creates an array of `nb_arrays` arrays of size `array_sizes` on device
    // contains only `nb_arrays_distinct` distinct elemnts
    ArrayBatch_withRepetition(const size_t array_sizes_args, const size_t nb_arrays_args, const size_t nb_arrays_distinct_arg=5, const bool should_initialize_data=false): array_sizes(array_sizes_args), nb_arrays(nb_arrays_args), nb_arrays_distinct(nb_arrays_distinct_arg)
    {
        // random number generator for the data generation
        std::random_device rd{};
        std::default_random_engine rng{rd()};
        // allocating the arrays
        rawPointer = new T*[nb_arrays];
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            rawPointer[i] = new T[array_sizes];
            if(should_initialize_data) fillArray(rawPointer[i], array_sizes, rng);
        }
        // allocates blocks of identical batch elements
        for(unsigned int i=nb_arrays_distinct; i<nb_arrays; i++)
        {
            const int ptr_index = (i * nb_arrays_distinct) / nb_arrays;
            rawPointer[i] = rawPointer[ptr_index];
        }
    }

    // deep copy constructor
    ArrayBatch_withRepetition(const ArrayBatch_withRepetition& arraybatch): ArrayBatch_withRepetition(arraybatch.array_sizes, arraybatch.nb_arrays, arraybatch.nb_arrays_distinct)
    {
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            std::memcpy(rawPointer[i], arraybatch.rawPointer[i], sizeof(T)*array_sizes);
        }
    }

    // releases the memory
    ~ArrayBatch_withRepetition()
    {
        // frees the batch elements
        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            delete[] rawPointer[i];
        }
        // free the array of batch elements
        delete[] rawPointer;
    }

    // computes the maximum relative distance between two arraybatch
    T distance(const ArrayBatch_withRepetition& arraybatch)
    {
        const T epsilon = 1e-15;
        T max_dist = 0.;

        for(unsigned int i=0; i<nb_arrays_distinct; i++)
        {
            T* v1 = rawPointer[i];
            const T* v2 = arraybatch.rawPointer[i];
            for(unsigned int j = 0; j < array_sizes; j++)
            {
                const T dist = std::abs(v1[i] - v2[i]) / (std::abs(v1[i]) + epsilon);
                if(dist > max_dist)  max_dist = dist;
            }
        }

        return max_dist;
    }
};