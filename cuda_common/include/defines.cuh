#ifndef CUDA_DEFINES_HEADER
#define CUDA_DEFINES_HEADER

#ifdef __CUDACC__
#define CUDA_HOST_CALLABLE __host__ __device__
#define HOST_CALLABLE __host__
#define CUDA_CALLABLE __device__
#else
#define CUDA_HOST_CALLABLE
#define HOST_CALLABLE
#define CUDA_CALLABLE
#endif

#endif
