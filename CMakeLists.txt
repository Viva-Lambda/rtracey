cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("RTracey" LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER "clang")

include_directories(
    "./include/"
)
add_library(AssimpLib SHARED IMPORTED)
set_property(TARGET 
    AssimpLib 
    PROPERTY IMPORTED_LOCATION 
    "/media/kaan/Data7510/GitProjects/assimp/build/bin/libassimp.so")

add_executable(
    rtracey.out
    "src/main.cu"
)
target_link_libraries(rtracey.out AssimpLib)

find_package(CUDA REQUIRED)
#find_package(Thrust REQUIRED)
#thrust_create_target(Thrust)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(
  CMAKE_CXX_FLAGS 
  "-lstdc++ -ldl -ggdb -Wall -Wextra -pedantic -Wextra-tokens
  -stdlib=libstdc++"
)

set(CMAKE_CUDA_FLAGS "-ccbin clang++ -G -g -gencode arch=compute_50,code=sm_50")

#target_compile_features(main PUBLIC cxx_std_11)

set_target_properties(
    rtracey.out
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
)
set_target_properties(rtracey.out
    PROPERTIES LINKER_LANGUAGE CUDA
)

install(TARGETS rtracey.out DESTINATION "${PROJECT_SOURCE_DIR}/bin")
