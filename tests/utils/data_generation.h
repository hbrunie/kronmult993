#pragma once
#include <random>

/*
 * fill a given array with data sampled from a normal distribution
 * uses RNG as a random generator
 */
template<typename T, typename RNG>
void fillArray(T a[], size_t const nb_element, RNG &gen)
{
    // distribution used to generate data
    std::normal_distribution<> d;

    // fills the array
    for (unsigned int i = 0; i < nb_element; i++)
    {
        a[i] = d(gen);
    }
}
