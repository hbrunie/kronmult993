cmake_minimum_required(VERSION 3.18)
project(kronmult)
#cmake_policy(SET CMP0074 NEW)

# add kronmult cpu implementation
add_subdirectory(kronmult_omp)

# add kronmult gpu implementation if cuda is available
find_package(CUDA)
if (CUDA_FOUND)
    add_subdirectory(kronmult_gpu)
endif()

# adds tests
include(CTest)
enable_testing()
add_subdirectory(tests)
