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

template <typename T>
void upload_thrust(thrust::device_ptr<T> &d_ptr, T *&h_ptr,
                   int count) {
  d_ptr = thrust::device_malloc<T>(count);
  for (int i = 0; i < count; i++)
    d_ptr[i] = h_ptr[i];
}
