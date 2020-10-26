#pragma once
//
#include <aabb.cuh>
#include <groupparam.cuh>
#include <minmax.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

template <HittableType h>
__host__ __device__ bool
bounding_box(const SceneObjects &s, float t0, float t1,
             Aabb &output_box, int prim_idx) {
  HittableParam hs = s.get_hparam(prim_idx);
  output_box =
      Aabb(min_vec<HITTABLE>(hs), max_vec<HITTABLE>(hs));
  return true;
}
template <>
__host__ __device__ bool
bounding_box<HITTABLE>(const SceneObjects &s, float t0,
                       float t1, Aabb &output_box,
                       int prim_idx) {
  HittableParam h = s.get_hparam(prim_idx);
  output_box =
      Aabb(min_vec<HITTABLE>(h), max_vec<HITTABLE>(h));
  return true;
}

template <GroupType g>
__host__ __device__ bool
bounding_box(const SceneObjects &s, float t0, float t1,
             Aabb &output_box, int group_idx) {
  return false;
}

template <>
__host__ __device__ bool
bounding_box<NONE_GRP>(const SceneObjects &s, float t0,
                       float t1, Aabb &output_box,
                       int group_idx) {
  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  int group_end = group_start + group_size;
  bool is_bounding = false;
  Aabb temp;
  bool first_box = true;
  for (int i = group_start; i < group_end; i++) {
    int prim_idx = i;
    is_bounding = bounding_box<HITTABLE>(
        s, t0, t1, output_box, prim_idx);
    if (is_bounding == false) {
      return false;
    }
    output_box = first_box
                     ? temp
                     : surrounding_box(output_box, temp);
    first_box = false;
    // center = output_box.center;
  }
  return is_bounding;
}

template <>
__host__ __device__ bool
bounding_box<BOX>(const SceneObjects &s, float t0, float t1,
                  Aabb &output_box, int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
template <>
__host__ __device__ bool bounding_box<CONSTANT_MEDIUM>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
template <>
__host__ __device__ bool
bounding_box<SIMPLE_MESH>(const SceneObjects &s, float t0,
                          float t1, Aabb &output_box,
                          int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
template <>
__host__ __device__ bool
bounding_box<OBJECT>(const SceneObjects &s, float t0,
                     float t1, Aabb &output_box,
                     int group_idx) {
  GroupType gtype =
      static_cast<GroupType>(s.gtypes[group_idx]);
  bool res = false;
  if (gtype == NONE_GRP) {
    res = bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                 group_idx);
  } else if (gtype == BOX) {
    res =
        bounding_box<BOX>(s, t0, t1, output_box, group_idx);
  } else if (gtype == CONSTANT_MEDIUM) {
    res = bounding_box<CONSTANT_MEDIUM>(
        s, t0, t1, output_box, group_idx);
  } else if (gtype == SIMPLE_MESH) {
    res = bounding_box<SIMPLE_MESH>(s, t0, t1, output_box,
                                    group_idx);
  }
  return res;
}

template <typename T>
__host__ __device__ bool bounding_box(const T &obj,
                                      Aabb &o) {
  return false;
}
template <>
__host__ __device__ bool
bounding_box<GroupParam>(const GroupParam &o, Aabb &obj) {
  obj =
      Aabb(min_vec<GroupParam>(o), max_vec<GroupParam>(o));
  return true;
}
