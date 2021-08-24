#include "utils/utils_cpu.h"
#include "utils/batch_size.h"
#include <chrono>
#include <iostream>
#include <kronmult.hpp>
#include <omp.h>

// change this to run the bench in another precision
using Number = double;

struct data_type{
    double time_ms_mean;
    double time_ms_total;
    int iter;
};

/*
 * runs a benchmark with the given parameters
 * `nb_distinct_outputs` modelizes the fact that most outputs are identical
 */
struct data_type runBench(int const degree, int const dimension, int const grid_level, std::string const benchName,
              int const nb_distinct_outputs = 5)
{
    // Kronmult parameters
    int const matrix_size  = degree;
    int const matrix_count = dimension;
    int const size_input   = pow_int(matrix_size, matrix_count);
    int const matrix_stride = 67; // large prime integer, modelize the fact that columns are not adjascent in memory
    int const batch_count = compute_batch_size(degree, dimension, grid_level, nb_distinct_outputs);
    //std::cout << benchName << " benchcase"
    //          << " batch_count:" << batch_count << " matrix_size:" << matrix_size
    //          << " matrix_count:" << matrix_count << " size_input:" << size_input
    //          << " nb_distinct_outputs:" << nb_distinct_outputs << std::endl;

    // allocates a problem
    // we do not put data in the vectors/matrices as it doesn't matter here
    //std::cout << "Starting allocation." << std::endl;
    ArrayBatch<Number> matrix_list_batched(matrix_size * matrix_stride, batch_count * matrix_count);
    ArrayBatch<Number> input_batched(size_input, batch_count);
    ArrayBatch<Number> workspace_batched(size_input, batch_count);
    ArrayBatch_withRepetition<Number> output_batched(size_input, batch_count, nb_distinct_outputs);

    int iter = 1;
    int iter_next = 2;
    // runs kronmult several times and displays the average runtime
    //std::cout << "Starting Kronmult" << std::endl;
    double time_ms_total = 0.;
    double time_ms_mean = 0.;
    cudaError errorCode;
    do {
        iter = iter_next ;
        auto start                = std::chrono::high_resolution_clock::now();
        for(int k =0; k<iter;k++)
            errorCode = kronmult_batched(
            matrix_count, matrix_size, matrix_list_batched.rawPointer, matrix_stride, input_batched.rawPointer,
            output_batched.rawPointer, workspace_batched.rawPointer, batch_count);
        auto stop         = std::chrono::high_resolution_clock::now();
        time_ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        time_ms_mean = time_ms_total / iter;
        iter_next *= 2;
    } while (time_ms_total < 50.);
    struct data_type data;
    data.time_ms_mean = time_ms_mean;
    data.time_ms_total = time_ms_total;
    data.iter = iter;
    return data;
}

/*
 * Runs benchmarks of increasing sizes and displays the results
 */
int main()
{
    std::cerr << "name(deg_dim_batchsize);degree;dimension;batchsize;level;time_ms_mean;time_ms_total;iter" << std::endl;
    // running the benchmarks
    std::vector<std::string> names;
    std::vector<long> times;
    for(int degree = 2; degree <= 10; degree++)
    {
        for(int dimension = 2; dimension <= 6; dimension++)
        {
            for(int level = 2; level <= 9; level++)
            {
                // run bench
                int const nb_distinct_outputs = 5;
                int const batch_count = compute_batch_size(degree, dimension, level, nb_distinct_outputs);
                //std::string name = "degree:" + std::to_string(degree) + " dimension:" + std::to_string(dimension)
                //                 + " level:" + std::to_string(level) + " batch-size:" + std::to_string(batch_count);
                std::string name = std::to_string(degree) + "_" + std::to_string(dimension)
                                 + "_" + std::to_string(batch_count);
                name += ";" + std::to_string(degree) + ";" + std::to_string(dimension)
                     + ";" + std::to_string(batch_count) + ";" + std::to_string(level);
                struct data_type data = runBench(degree, dimension, level, name, nb_distinct_outputs);
                std::cerr << name << ";"<<data.time_ms_mean << ";" << data.time_ms_total << ";" << data.iter << std::endl;
                // strore result
            }
        }
    }
}
