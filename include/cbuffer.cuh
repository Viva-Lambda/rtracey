#pragma once
#include <external.hpp>
#include <ray.cuh>
#include <vec3.cuh>

template <typename T>
cudaError_t alloc(T *&d_ptr, std::size_t size) {
  d_ptr = nullptr;
  return cudaMalloc((void **)&d_ptr, size);
}
template <typename T>
cudaError_t upload(T *&d_ptr, T *&h_ptr, int count) {
  alloc(d_ptr, count * sizeof(T));
  return cudaMemcpy((void *)d_ptr, (void *)h_ptr,
                    count * sizeof(T),
                    cudaMemcpyHostToDevice);
}
