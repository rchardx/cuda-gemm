set(__SRCS__)
set(__INCLUDE_DIRS__)
set(__LINK_LIBS__)

file(GLOB __TMP_SRCS__
    *.cu)
list(APPEND __SRCS__ ${__TMP_SRCS__})
unset(__TMP_SRCS__)

list(APPEND __INCLUDE_DIRS__ ${CUDAToolkit_INCLUDE_DIRS})
list(APPEND __LINK_LIBS__
    CUDA::cuda_driver CUDA::cudart CUDA::cublasLt CUDA::cupti)

add_executable(gemm ${__SRCS__})

target_include_directories(gemm PUBLIC
    ${__INCLUDE_DIRS__})

target_link_libraries(gemm PRIVATE
    ${__LINK_LIBS__})

set_property(TARGET gemm PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

unset(__SRCS__)
unset(__INCLUDE_DIRS__)
unset(__LINK_LIBS__)
