#pragma once
#include <group.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct GroupParam {
  // group params
  GroupType gtype;
  int group_size;
  int group_id;
  Primitive *prims;
  //
  float density;
  TextureParam tparam;

  __host__ __device__ GroupParam() {}
  __host__ __device__ GroupParam(Primitive *prm, int gsize,
                                 int gid, GroupType gtp,
                                 float d, TextureParam tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), tparam(tp) {}
  __host__ __device__ inline bool get(int i,
                                      Primitive &p) const {
    if (i >= 0 && i < group_size) {
      p = prims[i];
      return true;
    }
    return false;
  }
  __host__ __device__ Box to_box() const {
    Box b(prims);
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
