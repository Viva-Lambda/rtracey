#pragma once
#include "debug.hpp"
#include <external.hpp>
#include <ray.cuh>
#include <vec3.cuh>

template <typename T>
T *alloc(T *&d_ptr, std::size_t size, cudaError_t &err) {
  d_ptr = nullptr;
  err = cudaMalloc((void **)&(d_ptr), size);
  return d_ptr;
}
template <typename T>
T *upload(T *&d_ptr, T *&h_ptr, int count,
          cudaError_t &err) {
  d_ptr = alloc(d_ptr, count * sizeof(T), err);
  CUDA_CONTROL(err);
  err =
      cudaMemcpy((void *)d_ptr, (void *)h_ptr,
                 count * sizeof(T), cudaMemcpyHostToDevice);
  return d_ptr;
}

template <typename T>
void upload_thrust(thrust::device_ptr<T> &d_ptr, T *&h_ptr,
                   int count) {
  d_ptr = thrust::device_malloc<T>(count);
  for (int i = 0; i < count; i++)
    d_ptr[i] = h_ptr[i];
}
