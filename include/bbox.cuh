#pragma once
//
#include <aabb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <sceneshape.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

template <HittableType h>
__host__ __device__ bool
bounding_box(const SceneObjects &s, float t0, float t1,
             Aabb &output_box, int prim_idx) {
  return true;
}
template <>
__host__ __device__ bool
bounding_box<SPHERE_HIT>(const SceneObjects &s, float t0,
                         float t1, Aabb &output_box,
                         int prim_idx) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];
  output_box =
      Aabb(center - Vec3(radius), center + Vec3(radius));
  return true;
}

template <>
__host__ __device__ bool bounding_box<MOVING_SPHERE_HIT>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int prim_idx) {
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);
  Point3 center = (center1 + center2) / 2.0f;

  float radius = s.rads[prim_idx];
  output_box =
      Aabb(center - Vec3(radius), center + Vec3(radius));
  return true;
}
template <>
__host__ __device__ bool
bounding_box<TRIANGLE_HIT>(const SceneObjects &s, float t0,
                           float t1, Aabb &output_box,
                           int prim_idx) {
  Point3 p1(s.p1xs[prim_idx], s.p1ys[prim_idx],
            s.p1zs[prim_idx]);
  Point3 p2(s.p2xs[prim_idx], s.p2ys[prim_idx],
            s.p2zs[prim_idx]);
  Point3 p3(s.n1xs[prim_idx], s.n1ys[prim_idx],
            s.n1zs[prim_idx]);
  Point3 pmin = min_vec(p1, p2);
  pmin = min_vec(pmin, p3);
  Point3 pmax = max_vec(p1, p2);
  pmax = max_vec(pmax, p3);
  output_box = Aabb(pmin, pmax);
  return true;
}
template <>
__host__ __device__ bool bounding_box<XY_TRIANGLE_HIT>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int prim_idx) {
  return bounding_box<TRIANGLE_HIT>(s, t0, t1, output_box,
                                    prim_idx);
}
template <>
__host__ __device__ bool bounding_box<XZ_TRIANGLE_HIT>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int prim_idx) {
  return bounding_box<TRIANGLE_HIT>(s, t0, t1, output_box,
                                    prim_idx);
}
template <>
__host__ __device__ bool bounding_box<YZ_TRIANGLE_HIT>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int prim_idx) {
  return bounding_box<TRIANGLE_HIT>(s, t0, t1, output_box,
                                    prim_idx);
}

template <>
__host__ __device__ bool
bounding_box<RECT_HIT>(const SceneObjects &s, float t0,
                       float t1, Aabb &output_box,
                       int prim_idx) {
  float k = s.rads[prim_idx];
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p1ys[prim_idx];
  float b0 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Vec3 anormal = Vec3(s.n1xs[prim_idx], s.n1ys[prim_idx],
                      s.n1zs[prim_idx]);
  AxisInfo ax = AxisInfo(anormal);

  Point3 p1, p2;
  // choose points with axis
  switch (ax.notAligned) {
  case 2: {
    p1 = Point3(a0, b0, k - 0.0001);
    p2 = Point3(a1, b1, k + 0.0001);
    break;
  }
  case 1: {
    p1 = Point3(a0, k - 0.0001, b0);
    p2 = Point3(a1, k + 0.0001, b1);
    break;
  }
  case 0: {
    p1 = Point3(k - 0.0001, a0, b0);
    p2 = Point3(k + 0.0001, a1, b1);
    break;
  }
  }
  output_box = Aabb(p1, p2);
  return true;
}

template <>
__host__ __device__ bool
bounding_box<XY_RECT_HIT>(const SceneObjects &s, float t0,
                          float t1, Aabb &output_box,
                          int prim_idx) {
  return bounding_box<RECT_HIT>(s, t0, t1, output_box,
                                prim_idx);
}

template <>
__host__ __device__ bool
bounding_box<XZ_RECT_HIT>(const SceneObjects &s, float t0,
                          float t1, Aabb &output_box,
                          int prim_idx) {
  return bounding_box<RECT_HIT>(s, t0, t1, output_box,
                                prim_idx);
}

template <>
__host__ __device__ bool
bounding_box<YZ_RECT_HIT>(const SceneObjects &s, float t0,
                          float t1, Aabb &output_box,
                          int prim_idx) {
  return bounding_box<RECT_HIT>(s, t0, t1, output_box,
                                prim_idx);
}
template <>
__host__ __device__ bool
bounding_box<HITTABLE>(const SceneObjects &s, float t0,
                       float t1, Aabb &output_box,
                       int prim_idx) {
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype);
  bool res = false;
  switch (htype) {
  case NONE_HIT: {
    break;
  }
  case SPHERE_HIT: {
    res = bounding_box<SPHERE_HIT>(s, t0, t1, output_box,
                                   prim_idx);
    break;
  }
  case MOVING_SPHERE_HIT: {
    res = bounding_box<MOVING_SPHERE_HIT>(
        s, t0, t1, output_box, prim_idx);
    break;
  }
  case XY_RECT_HIT: {
    res = bounding_box<XY_RECT_HIT>(s, t0, t1, output_box,
                                    prim_idx);
    break;
  }
  case XZ_RECT_HIT: {
    res = bounding_box<XZ_RECT_HIT>(s, t0, t1, output_box,
                                    prim_idx);
    break;
  }
  case YZ_RECT_HIT: {
    res = bounding_box<YZ_RECT_HIT>(s, t0, t1, output_box,
                                    prim_idx);
    break;
  }
  case YZ_TRIANGLE_HIT: {
    res = bounding_box<YZ_TRIANGLE_HIT>(
        s, t0, t1, output_box, prim_idx);
    break;
  }
  case XZ_TRIANGLE_HIT: {
    res = bounding_box<XZ_TRIANGLE_HIT>(
        s, t0, t1, output_box, prim_idx);
    break;
  }
  case XY_TRIANGLE_HIT: {
    res = bounding_box<XY_TRIANGLE_HIT>(
        s, t0, t1, output_box, prim_idx);
    break;
  }
  }
  return res;
}

template <GroupType g>
__host__ __device__ bool bounding_box(const SceneObjects &s,
                                      float t0, float t1,
                                      Aabb &output_box) {
  return false;
}
template <NONE_GRP>
__host__ __device__ bool
bounding_box(const SceneObjects &s, float t0, float t1,
             Aabb &output_box, int group_idx) {

  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  bool is_bounding = false;
  Aabb temp;
  bool first_box = true;
  for (int i = group_start; i < group_size; i++) {
    int prim_idx = i;
    int htype_ = s.htypes[prim_idx];
    res = bounding_box<HITTABLE>(s, t0, t1, output_box,
                                 prim_idx);
    if (is_bounding == false) {
      return false;
    }
    output_box = first_box
                     ? temp
                     : surrounding_box(output_box, temp);
    first_box = false;
    // center = output_box.center;
  }
  return res;
}
template <>
__host__ __device__ bool
bounding_box<BOX_GRP>(const SceneObjects &s, float t0,
                      float t1, Aabb &output_box,
                      int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
template <>
__host__ __device__ bool bounding_box<CONSTANT_MEDIUM_GRP>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
template <>
__host__ __device__ bool bounding_box<SIMPLE_MESH_GRP>(
    const SceneObjects &s, float t0, float t1,
    Aabb &output_box, int group_idx) {
  return bounding_box<NONE_GRP>(s, t0, t1, output_box,
                                group_idx);
}
