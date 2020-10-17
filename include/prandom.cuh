#pragma once
//
#include <aabb.cuh>
#include <hit.cuh>
#include <onb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <sceneshape.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

template <HittableType h>
__device__ Vec3 random(const SceneObjects &s,
                       const Point3 &o, curandState *loc,
                       int prim_idx) {
  return random_vec(loc);
}

template <>
__device__ Vec3 random<SPHERE_HIT>(const SceneObjects &s,
                                   const Point3 &o,
                                   curandState *loc,
                                   int prim_idx) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];

  Vec3 direction = center - o;
  auto distance_squared = direction.squared_length();
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, loc));
}

template <>
__device__ Vec3 random<MOVING_SPHERE_HIT>(
    const SceneObjects &s, const Point3 &o,
    curandState *loc, int prim_idx) {
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  float radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];
  float tdiff = time1 - time0;
  Point3 scenter = MovingSphere::mcenter(
      center1, center2, time0, time1, tdiff);

  Vec3 direction = scenter - o;
  auto distance_squared = direction.squared_length();
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, loc));
}
template <>
__device__ Vec3 random<TRIANGLE_HIT>(const SceneObjects &s,
                                     const Point3 &o,
                                     curandState *loc,
                                     int prim_idx) {
  //
  Point3 p1(s.p1xs[prim_idx], s.p1ys[prim_idx],
            s.p1zs[prim_idx]);
  Point3 p2(s.p2xs[prim_idx], s.p2ys[prim_idx],
            s.p2zs[prim_idx]);
  Point3 p3(s.n1xs[prim_idx], s.n1ys[prim_idx],
            s.n1zs[prim_idx]);

  // from A. Glassner, Graphics Gems, 1995, p. 24
  float t = curand_uniform(loc);
  float s = curand_uniform(loc);
  auto a = 1 - sqrt(t);
  auto b = (1 - s) * sqrt(t);
  auto c = s * sqrt(t);
  auto random_point = a * p1 + b * p2 + c * p3;
  return random_point - o;
}

template <>
__device__ Vec3 random<XY_TRIANGLE_HIT>(
    const SceneObjects &s, const Point3 &o,
    curandState *loc, int prim_idx) {
  return random<TRIANGLE_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<XZ_TRIANGLE_HIT>(
    const SceneObjects &s, const Point3 &o,
    curandState *loc, int prim_idx) {
  return random<TRIANGLE_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<YZ_TRIANGLE_HIT>(
    const SceneObjects &s, const Point3 &o,
    curandState *loc, int prim_idx) {
  return random<TRIANGLE_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<RECT_HIT>(const SceneObjects &s,
                                 const Point3 &o,
                                 curandState *loc,
                                 int prim_idx) {
  float k = s.rads[prim_idx];
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p1ys[prim_idx];
  float b0 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Point3 random_point = Point3(random_float(loc, a0, a1), k,
                               random_float(loc, b0, b1));
  return random_point - o;
}

template <>
__device__ Vec3 random<XY_RECT_HIT>(const SceneObjects &s,
                                    const Point3 &o,
                                    curandState *loc,
                                    int prim_idx) {
  return random<RECT_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<XZ_RECT_HIT>(const SceneObjects &s,
                                    const Point3 &o,
                                    curandState *loc,
                                    int prim_idx) {
  return random<RECT_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<YZ_RECT_HIT>(const SceneObjects &s,
                                    const Point3 &o,
                                    curandState *loc,
                                    int prim_idx) {
  return random<RECT_HIT>(s, o, loc, prim_idx);
}
template <>
__device__ Vec3 random<HITTABLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 curandState *loc,
                                 int prim_idx) {
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype);
  Vec3 c(0.0f);
  if (htype == SPHERE_HIT) {
    c = random<SPHERE_HIT>(s, o, loc, prim_idx);
  } else if (htype == MOVING_SPHERE_HIT) {
    c = random<MOVING_SPHERE_HIT>(s, o, loc, prim_idx);
  } else if (htype == XY_RECT_HIT) {
    c = random<XY_RECT_HIT>(s, o, loc, prim_idx);
  } else if (htype == XZ_RECT_HIT) {
    c = random<XZ_RECT_HIT>(s, o, loc, prim_idx);
  } else if (htype == YZ_RECT_HIT) {
    c = random<YZ_RECT_HIT>(s, o, loc, prim_idx);
  } else if (htype == YZ_TRIANGLE_HIT) {
    c = random<YZ_TRIANGLE_HIT>(s, o, loc, prim_idx);
  } else if (htype == XZ_TRIANGLE_HIT) {
    c = random<XZ_TRIANGLE_HIT>(s, o, loc, prim_idx);
  } else if (htype == XY_TRIANGLE_HIT) {
    c = random<XY_TRIANGLE_HIT>(s, o, loc, prim_idx);
  }
  return c;
}
