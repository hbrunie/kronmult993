
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O3 -g")

# define adds OpenMP flags
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (USE_MKL)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message("-- OpenMP detected ---")
else ()
    message(WARNING "Building Kronmult without OpenMP support.")
endif ()

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

set(KRONSRC
        kronmult_openmp.hpp
        kronmult_gpu.hpp
        # openMP
        openmp/kronmult.hpp
        openmp/linear_algebra.hpp
        # GPU
        gpu/kronmult.hpp
        gpu/gpu_operations.hpp)

if (USE_GPU)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fPIC --keep-device-functions")
    if (USE_MKL)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch sm_60 --compiler-options -fopenmp -fPIC --keep-device-functions")
    endif ()
    set_source_files_properties(${KRONSRC} PROPERTIES LANGUAGE CUDA)
endif ()

add_library(kron SHARED ${KRONSRC})
set_target_properties(kron PROPERTIES LINKER_LANGUAGE CXX)

if (USE_GPU)
    set_target_properties(kron PROPERTIES CUDA_ARCHITECTURES OFF)
else()
    set_target_properties(kron PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS OFF)
endif ()
target_compile_features(kron INTERFACE cxx_std_17)## INTERFACE or PUBLIC?
target_include_directories(kron INTERFACE ./) ## INTERFACE or PUBLIC?

#set(TESTSRC tests/test_kronmult6_xbatched.cpp)

#if (USE_GPU)
#    set_source_files_properties(${TESTSRC} PROPERTIES LANGUAGE CUDA)
#endif ()
