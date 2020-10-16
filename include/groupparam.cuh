#pragma once
#include <group.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct GroupParam {
  // group params
  const int *gtype;
  const int *group_size;
  const int *group_id;
  const Primitive *prims;
  //
  const float *density;

  const TextureParam tparam;

  __host__ __device__ GroupParam() {}
  __host__ __device__ GroupParam(const Primitive *prm,
                                 const int *gsize,
                                 const int *gid,
                                 const int *gtp,
                                 const float *d,
                                 const TextureParam &tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), tparam(tp) {}
  __host__ __device__ Primitive get(int i) const {
    if (i <= 0) {
      return prims[0];
    } else if (i < (*group_size)) {
      return prims[i];
    } else {
      return prims[(*group_size) - 1];
    }
  }
  __host__ __device__ Box to_box() const {
    Box b(prims, group_size);
    return b;
  }
  __host__ __device__ SimpleMesh to_simple_mesh() const {
    SimpleMesh sm(prims, group_size);
    return sm;
  }
  __host__ __device__ ConstantMedium
  to_constant_medium() const {
    ConstantMedium cm(prims, density, tparam, group_size);
    return cm;
  }
};
