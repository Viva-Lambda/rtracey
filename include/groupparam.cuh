#pragma once
#include <group.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct GroupParam {
  // group params
  const int gtype;
  const int group_size;
  const int group_id;
  Primitive *prims;
  //
  const float density;

  const TextureParam tparam;

  __host__ __device__ GroupParam()
      : gtype(0), group_size(0), group_id(0), density(0.0f),
        prims(nullptr) {}
  __host__ __device__ GroupParam(
      Primitive *prm, const int gsize, const int gid,
      const int gtp, const float d, const TextureParam &tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), tparam(tp) {}
  __host__ __device__ Primitive get(int i) const {
    if (i <= 0) {
      return prims[0];
    } else if (i < group_size) {
      return prims[i];
    } else {
      return prims[group_size - 1];
    }
  }
};
