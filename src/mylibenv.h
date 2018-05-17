
#define CUDA_BLOCK_SIZE 256
#define CUDA_BLOCK_SIZE2 16

#define CUDA_SAFE_CALL(call) \
do {                         \
    cudaError_t err = call;  \
    if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", \
                 __FILE__, __LINE__, cudaGetErrorString(err) );      \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

// Round down to the nearest multiple of n
#define ROUNDDOWN(a, n)                                         \
({                                                              \
        uint32_t __a = (uint32_t) (a);                          \
        (typeof(a)) (__a - __a % (n));                          \
})

// Round up to the nearest multiple of n
#define ROUNDUP(a, n)                                           \
({                                                              \
        uint32_t __n = (uint32_t) (n);                          \
        (typeof(a)) (ROUNDDOWN((uint32_t) (a) + __n - 1, __n)); \
})

