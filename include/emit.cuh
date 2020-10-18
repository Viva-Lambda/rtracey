#pragma once
// material emit fns
#include <attenuation.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <MaterialType m>
__device__ Color emitted(const SceneObjects &s,
                         const HitRecord &rec) {
  return Color(0.0f);
}

template <>
__device__ Color emitted<DIFFUSE_LIGHT>(
    const SceneObjects &s, const HitRecord &rec) {
  return color_value<TEXTURE>(s, rec);
}
template <>
__device__ Color emitted<MATERIAL>(const SceneObjects &s,
                                   const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Color res(0.0f);
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
  if (mtype == DIFFUSE_LIGHT) {
    res = emitted<DIFFUSE_LIGHT>(s, rec);
  }
  return res;
}

// cpu test function
__host__ __device__ void
emit_material(const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Color res(0.0f);
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
}
//
