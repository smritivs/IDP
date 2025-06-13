cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pinn-cuda LANGUAGES CXX CUDA)

# find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(HEADER_FILES ./inc/kan.h ./inc/fourier.h)

add_executable(ex ./src/kan.cu ${HEADER_FILES})
target_include_directories(ex PRIVATE ./inc)
# target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET ex PROPERTY CXX_STANDARD 17)
