set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# compilation flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -march=native")

# adds OpenMP flags and lib
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("-- OpenMP detected ---")
else ()
    message(WARNING "Building Kronmult without OpenMP support.")
endif ()

# adds BLAS lib
# define KRONMULT_USE_BLAS if BLAS is found
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if (BLAS_FOUND)
    add_compile_definitions(KRONMULT_USE_BLAS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
    message("-- BLAS detected ---")
else()
    message(WARNING "Building Kronmult without BLAS support.")
endif()

# full benchmark
add_executable(kronmult_fullbench kronmult_fullbench.cpp)
target_link_libraries(kronmult_fullbench PUBLIC kron ${BLAS_LIBRARIES})
target_include_directories(kronmult_fullbench PUBLIC ${CMAKE_SOURCE_DIR}/kronmult)
add_test(NAME kronmult_fullbench COMMAND ./kronmult_fullbench)

# test
add_executable(test_kronmult test_kronmult.cpp)
target_link_libraries(test_kronmult PUBLIC kron ${BLAS_LIBRARIES})
target_include_directories(test_kronmult  PUBLIC ${CMAKE_SOURCE_DIR}/kronmult)
add_test(NAME test_kronmult COMMAND ./test_kronmult)

# benchmark
add_executable(kronmult_bench kronmult_bench.cpp)
target_link_libraries(kronmult_bench PUBLIC kron ${BLAS_LIBRARIES})
target_include_directories(kronmult_bench PUBLIC ${CMAKE_SOURCE_DIR}/kronmult)
add_test(NAME kronmult_bench COMMAND ./kronmult_bench)
