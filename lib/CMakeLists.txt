cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(cufhog)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT WIN32)
ADD_DEFINITIONS("-std=c++0x -O3")
endif(NOT WIN32)

include_directories(./)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)


set(sourcefile cufhog.cu cufhog.h fhog_api.cu fhog_api.h my_util.cu my_util.h mylibenv.h)
CUDA_ADD_LIBRARY(cufhog SHARED ${sourcefile})

# We need to explicitly state that we need all CUDA files in the 
# cuhog library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( cufhog
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

link_directories(/home/lvfai/git-repo/KCFcpp/lib)
set(testfile test.cpp fhog_api.h)
add_executable(demo ${testfile})
target_link_libraries(demo cufhog ${OpenCV_LIBS} /usr/local/cuda/lib64/libcublas.so)
