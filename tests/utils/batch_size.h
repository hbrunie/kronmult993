#pragma once

/*
 * returns the batch size to be used with the given parameters
 * NOTE the use of long long is needed to avoid overflow in intermediate steps
 * TODO find Asgard's formula for batch count, current one can generates batches too large to be allocated without the min
 */
int compute_batch_size(int const degree, int const dimension, int const grid_level, int const nb_distinct_outputs)
{
    // Kronmult parameters
    int const matrix_size  = degree;
    int const matrix_count = dimension;
    int const size_input   = pow_int(matrix_size, matrix_count);
    // upper bound to insure that allocation is possible
    // nb_elements = batch_count * size_input * (2 + matrix_count*matrix_size*matrix_size) + nb_distinct_outputs*size_input;
    long long const max_element_number = 395000000000;
    long long const max_batch_count = (max_element_number - nb_distinct_outputs * size_input) / static_cast<long long>(size_input * (2 + matrix_count * matrix_size * matrix_size));
    // formula with theorical batch count
    long long const formula_batch_count = pow_int(2, grid_level) * pow_int(grid_level, std::min(1, dimension - 1));
    return std::min(max_batch_count, formula_batch_count);
}