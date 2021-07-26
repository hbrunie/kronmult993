cmake_minimum_required(VERSION 3.8)

# compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

#----------------------------------------------------------------------------------------
# CPU

# adds OpenMP flags and lib
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "Using kronmult_omp without OpenMP support.")
endif ()

# adds BLAS lib
# define KRONMULT_USE_BLAS if BLAS is found
find_package(BLAS)
if (BLAS_FOUND)
    add_compile_definitions(KRONMULT_USE_BLAS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
else ()
    message(WARNING "Using kronmult_omp without BLAS support.")
endif ()

# test
add_executable(kronmult_test_cpu kronmult_test.cpp)
target_link_libraries(kronmult_test_cpu PUBLIC kronmult_omp ${BLAS_LIBRARIES})
add_test(NAME kronmult_test_cpu COMMAND ./kronmult_test)

# benchmark
add_executable(kronmult_bench kronmult_bench.cpp)
target_link_libraries(kronmult_bench PUBLIC kronmult_omp ${BLAS_LIBRARIES})
add_test(NAME kronmult_bench COMMAND ./kronmult_bench)

#----------------------------------------------------------------------------------------
# GPU

# add CUDA lib
find_package(CUDA)
if (CUDA_FOUND)
    enable_language(CUDA)
    # test
    add_executable(kronmult_test_gpu kronmult_test_gpu.cpp)
    target_include_directories(kronmult_test_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kronmult_test_gpu PUBLIC kronmult_gpu ${CUDA_LIBRARIES} cublas)
    add_test(NAME kronmult_test_gpu COMMAND ./kronmult_test_gpu)
    # benchmark
    add_executable(kronmult_bench_gpu kronmult_bench_gpu.cpp)
    target_include_directories(kronmult_bench_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kronmult_bench_gpu PUBLIC kronmult_gpu ${CUDA_LIBRARIES} cublas)
    add_test(NAME kronmult_bench_gpu COMMAND ./kronmult_bench_gpu)

    add_executable(kronmult_fullbench_gpu kronmult_fullbench_gpu.cpp)
    target_include_directories(kronmult_fullbench_gpu PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(kronmult_fullbench_gpu PUBLIC kronmult_gpu ${CUDA_LIBRARIES} cublas)
    add_test(NAME kronmult_fullbench_gpu COMMAND ./kronmult_fullbench_gpu)
else ()
    message(WARNING "You cannot use kronmult_gpu without CUDA.")
endif ()
