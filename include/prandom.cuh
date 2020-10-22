#pragma once
//
#include <aabb.cuh>
#include <hit.cuh>
#include <onb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

template <HittableType h>
__device__ Vec3 random(const SceneObjects &s,
                       const Point3 &o, int prim_idx) {
  return random_vec(s.rand);
}

template <>
__device__ Vec3 random<SPHERE>(const SceneObjects &s,
                               const Point3 &o,
                               int prim_idx) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];

  Vec3 direction = center - o;
  auto distance_squared = direction.squared_length();
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, s.rand));
}

template <>
__device__ Vec3 random<MOVING_SPHERE>(const SceneObjects &s,
                                      const Point3 &o,
                                      int prim_idx) {
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  float radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];
  float tdiff = time1 - time0;
  Point3 scenter =
      moving_center(center1, center2, time0, time1, tdiff);

  Vec3 direction = scenter - o;
  auto distance_squared = direction.squared_length();
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, s.rand));
}
template <>
__device__ Vec3 random<TRIANGLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx) {
  //
  Point3 p1(s.p1xs[prim_idx], s.p1ys[prim_idx],
            s.p1zs[prim_idx]);
  Point3 p2(s.p2xs[prim_idx], s.p2ys[prim_idx],
            s.p2zs[prim_idx]);
  Point3 p3(s.n1xs[prim_idx], s.n1ys[prim_idx],
            s.n1zs[prim_idx]);

  // from A. Glassner, Graphics Gems, 1995, p. 24
  float t = curand_uniform(s.rand);
  float ss = curand_uniform(s.rand);
  auto a = 1 - sqrt(t);
  auto b = (1 - ss) * sqrt(t);
  auto c = ss * sqrt(t);
  auto random_point = a * p1 + b * p2 + c * p3;
  return random_point - o;
}

template <>
__device__ Vec3 random<XY_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return random<TRIANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<XZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return random<TRIANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<YZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return random<TRIANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<RECTANGLE>(const SceneObjects &s,
                                  const Point3 &o,

                                  int prim_idx) {
  float k = s.rads[prim_idx];
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p1ys[prim_idx];
  float b0 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Point3 random_point =
      Point3(random_float(s.rand, a0, a1), k,
             random_float(s.rand, b0, b1));
  return random_point - o;
}

template <>
__device__ Vec3 random<XY_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return random<RECTANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<XZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return random<RECTANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<YZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return random<RECTANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<HITTABLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx) {
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype_);
  Vec3 c(0.0f);
  if (htype == SPHERE) {
    c = random<SPHERE>(s, o, prim_idx);
  } else if (htype == MOVING_SPHERE) {
    c = random<MOVING_SPHERE>(s, o, prim_idx);
  } else if (htype == XY_RECT) {
    c = random<XY_RECT>(s, o, prim_idx);
  } else if (htype == XZ_RECT) {
    c = random<XZ_RECT>(s, o, prim_idx);
  } else if (htype == YZ_RECT) {
    c = random<YZ_RECT>(s, o, prim_idx);
  } else if (htype == YZ_TRIANGLE) {
    c = random<YZ_TRIANGLE>(s, o, prim_idx);
  } else if (htype == XZ_TRIANGLE) {
    c = random<XZ_TRIANGLE>(s, o, prim_idx);
  } else if (htype == XY_TRIANGLE) {
    c = random<XY_TRIANGLE>(s, o, prim_idx);
  }
  return c;
}

template <GroupType g>
__device__ Vec3 random(const SceneObjects &s,
                       const Point3 &o, int group_idx) {
  return random_vec(s.rand);
}
template <>
__device__ Vec3 random<NONE_GRP>(const SceneObjects &s,
                                 const Point3 &o,
                                 int group_idx) {
  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  int obj_index =
      random_int(s.rand, group_start, group_size - 1);
  return random<HITTABLE>(s, o, obj_index);
}
template <>
__device__ Vec3 random<BOX>(const SceneObjects &s,
                            const Point3 &o,
                            int group_idx) {
  return random<NONE_GRP>(s, o, group_idx);
}
template <>
__device__ Vec3 random<CONSTANT_MEDIUM>(
    const SceneObjects &s, const Point3 &o, int group_idx) {
  return random<NONE_GRP>(s, o, group_idx);
}
template <>
__device__ Vec3 random<SIMPLE_MESH>(const SceneObjects &s,
                                    const Point3 &o,
                                    int group_idx) {
  return random<NONE_GRP>(s, o, group_idx);
}
