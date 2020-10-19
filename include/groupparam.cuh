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

  const TextureType ttype;
  const float tp1x, tp1y, tp1z;
  const float scale;
  const int width, height, bytes_per_pixel, index;

  __host__ __device__ GroupParam()
      : gtype(0), group_size(0), group_id(0), density(0.0f),
        prims(nullptr), width(0), height(0),
        bytes_per_pixel(0), index(0), tp1x(0), tp1y(0),
        tp1z(0), scale(0), ttype(NONE_TEXTURE) {}
  __host__ __device__ GroupParam(
      Primitive *prm, const int gsize, const int gid,
      const int gtp, const float d, const TextureParam &tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), width(tp.width),
        height(tp.height),
        bytes_per_pixel(tp.bytes_per_pixel),
        index(tp.index), tp1x(tp.tp1x), tp1y(tp.tp1y),
        tp1z(tp.tp1z), scale(tp.scale), ttype(tp.ttype) {}
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
