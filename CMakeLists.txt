cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

set(CUDA_Include /usr/local/cuda/include)
#set(SOURCE src/cufftTool.h src/cufftTool.cpp ${CUDA_Include}/cufft.h ${CUDA_Include}/cuda_runtime.h)
set(CUDA_LIBS /usr/local/cuda/lib64/libcudart.so /usr/local/cuda/lib64/libcufft.so /usr/local/cuda/lib64/libcublas.so)

include_directories(src) 
link_directories(/home/lvfai/git-repo/KCFcpp/src)
FILE(GLOB_RECURSE sourcefiles src/*.cpp src/*.hpp src/*.h ${CUDA_Include}/cufft.h ${CUDA_Include}/cuda_runtime.h ${CUDA_Include}/cuda.h ${CUDA_Include}/cublas_v2.h)
#add_executable( KCF ${sourcefiles} )
#target_link_libraries( KCF ${OpenCV_LIBS})

add_executable(KCF ${sourcefiles})
target_include_directories(KCF PRIVATE ${CUDA_Include})
target_link_libraries( KCF ${OpenCV_LIBS} ${CUDA_LIBS} libcufhog.so)


