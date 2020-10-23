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
                       const Point3 &o, int prim_idx,
                       curandState *loc) {
  return random_vec(loc);
}
template <HittableType h>
__host__ Vec3 h_random(const SceneObjects &s,
                       const Point3 &o, int prim_idx) {
  return h_random_vec();
}
__host__ __device__ void
rand_sphere(const SceneObjects &s, const Point3 &o,
            int prim_idx, float &dsquared, Vec3 &direction,
            float &radius) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  radius = s.rads[prim_idx];
  direction = center - o;
  dsquared = direction.squared_length();
}

template <>
__device__ Vec3 random<SPHERE>(const SceneObjects &s,
                               const Point3 &o,
                               int prim_idx,
                               curandState *loc) {
  float radius;
  float distance_squared;
  Vec3 direction;
  rand_sphere(s, o, prim_idx, distance_squared, direction,
              radius);
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, loc));
}
template <>
__host__ Vec3 h_random<SPHERE>(const SceneObjects &s,
                               const Point3 &o,
                               int prim_idx) {
  //
  float radius;
  float distance_squared;
  Vec3 direction;
  rand_sphere(s, o, prim_idx, distance_squared, direction,
              radius);
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      h_random_to_sphere(radius, distance_squared));
}

__host__ __device__ void
rand_msphere(const SceneObjects &s, const Point3 &o,
             int prim_idx, float &dsquared, Vec3 &direction,
             float &radius) {
  //
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];
  float tdiff = time1 - time0;
  Point3 scenter =
      moving_center(center1, center2, time0, time1, tdiff);

  direction = scenter - o;
  float dsquared = direction.squared_length();
}

template <>
__device__ Vec3 random<MOVING_SPHERE>(const SceneObjects &s,
                                      const Point3 &o,
                                      int prim_idx,
                                      curandState *loc) {
  float radius;
  float distance_squared;
  Vec3 direction;
  rand_msphere(s, o, prim_idx, distance_squared, direction,
               radius);
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      random_to_sphere(radius, distance_squared, loc));
}

template <>
__host__ Vec3 h_random<MOVING_SPHERE>(const SceneObjects &s,
                                      const Point3 &o,
                                      int prim_idx) {

  float radius;
  float distance_squared;
  Vec3 direction;
  rand_msphere(s, o, prim_idx, distance_squared, direction,
               radius);
  Onb uvw;
  uvw.build_from_w(direction);
  return uvw.local(
      h_random_to_sphere(radius, distance_squared));
}
__host__ __device__ Vec3
random_triangle(const SceneObjects &s, const Point3 &o,
                int prim_idx, float t, float ss) {
  //
  Point3 p1(s.p1xs[prim_idx], s.p1ys[prim_idx],
            s.p1zs[prim_idx]);
  Point3 p2(s.p2xs[prim_idx], s.p2ys[prim_idx],
            s.p2zs[prim_idx]);
  Point3 p3(s.n1xs[prim_idx], s.n1ys[prim_idx],
            s.n1zs[prim_idx]);

  // from A. Glassner, Graphics Gems, 1995, p. 24
  auto a = 1 - sqrt(t);
  auto b = (1 - ss) * sqrt(t);
  auto c = ss * sqrt(t);
  auto random_point = a * p1 + b * p2 + c * p3;
  return random_point - o;
}
template <>
__device__ Vec3 random<TRIANGLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx,
                                 curandState *loc) {
  //
  // from A. Glassner, Graphics Gems, 1995, p. 24
  float t = curand_uniform(loc);
  float ss = curand_uniform(loc);
  return random_triangle(s, o, prim_idx, t, ss);
}
template <>
__host__ Vec3 h_random<TRIANGLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx) {
  // from A. Glassner, Graphics Gems, 1995, p. 24
  float t = hrandf();
  float ss = hrandf();
  return random_triangle(s, o, prim_idx, t, ss);
}

template <>
__device__ Vec3 random<XY_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx,
                                    curandState *loc) {
  return random<TRIANGLE>(s, o, prim_idx, loc);
}
template <>
__host__ Vec3 h_random<XY_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return h_random<TRIANGLE>(s, o, prim_idx);
}

template <>
__device__ Vec3 random<XZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx,
                                    curandState *loc) {
  return random<TRIANGLE>(s, o, prim_idx, loc);
}
template <>
__host__ Vec3 h_random<XZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return h_random<TRIANGLE>(s, o, prim_idx);
}
template <>
__device__ Vec3 random<YZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx,
                                    curandState *loc) {
  return random<TRIANGLE>(s, o, prim_idx, loc);
}
template <>
__host__ Vec3 h_random<YZ_TRIANGLE>(const SceneObjects &s,
                                    const Point3 &o,
                                    int prim_idx) {
  return h_random<TRIANGLE>(s, o, prim_idx);
}
__host__ __device__ Vec3 random_rect(const SceneObjects &s,
                                     const Point3 &o,

                                     int prim_idx, float t0,
                                     float t1) {
  //
  float k = s.p1zs[prim_idx];
  Vec3 anormal = Vec3(s.n1xs[prim_idx], s.n1ys[prim_idx],
                      s.n1zs[prim_idx]);
  Point3 rand_point;
  if (anormal.x() == 1) {
    rand_point = Point3(k, t0, t1);
  } else if (anormal.y() == 1) {
    rand_point = Point3(t0, k, t1);
  } else {
    rand_point = Point3(t0, t1, k);
  }
  return rand_point - o;
}

template <>
__device__ Vec3 random<RECTANGLE>(const SceneObjects &s,
                                  const Point3 &o,

                                  int prim_idx,
                                  curandState *loc) {
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p2xs[prim_idx];
  float b0 = s.p1ys[prim_idx];
  float b1 = s.p2ys[prim_idx];
  float t0 = random_float(loc, a0, a1);
  float t1 = random_float(loc, b0, b1);
  return random_rect(s, o, prim_idx, t0, t1);
}
template <>
__host__ Vec3 h_random<RECTANGLE>(const SceneObjects &s,
                                  const Point3 &o,

                                  int prim_idx) {
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p2xs[prim_idx];
  float b0 = s.p1ys[prim_idx];
  float b1 = s.p2ys[prim_idx];
  float t0 = h_random_float(a0, a1);
  float t1 = h_random_float(b0, b1);
  return random_rect(s, o, prim_idx, t0, t1);
}

template <>
__device__ Vec3 random<XY_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx,
                                curandState *loc) {
  return random<RECTANGLE>(s, o, prim_idx, loc);
}
template <>
__device__ Vec3 random<XZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx,
                                curandState *loc) {
  return random<RECTANGLE>(s, o, prim_idx, loc);
}
template <>
__device__ Vec3 random<YZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx,
                                curandState *loc) {
  return random<RECTANGLE>(s, o, prim_idx, loc);
}
template <>
__host__ Vec3 h_random<XY_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return h_random<RECTANGLE>(s, o, prim_idx);
}
template <>
__host__ Vec3 h_random<XZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return h_random<RECTANGLE>(s, o, prim_idx);
}
template <>
__host__ Vec3 h_random<YZ_RECT>(const SceneObjects &s,
                                const Point3 &o,
                                int prim_idx) {
  return h_random<RECTANGLE>(s, o, prim_idx);
}

template <>
__device__ Vec3 random<HITTABLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx,
                                 curandState *loc) {
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype_);
  Vec3 c(0.0f);
  if (htype == SPHERE) {
    c = random<SPHERE>(s, o, prim_idx, loc);
  } else if (htype == MOVING_SPHERE) {
    c = random<MOVING_SPHERE>(s, o, prim_idx, loc);
  } else if (htype == XY_RECT) {
    c = random<XY_RECT>(s, o, prim_idx, loc);
  } else if (htype == XZ_RECT) {
    c = random<XZ_RECT>(s, o, prim_idx, loc);
  } else if (htype == YZ_RECT) {
    c = random<YZ_RECT>(s, o, prim_idx, loc);
  } else if (htype == YZ_TRIANGLE) {
    c = random<YZ_TRIANGLE>(s, o, prim_idx, loc);
  } else if (htype == XZ_TRIANGLE) {
    c = random<XZ_TRIANGLE>(s, o, prim_idx, loc);
  } else if (htype == XY_TRIANGLE) {
    c = random<XY_TRIANGLE>(s, o, prim_idx, loc);
  }
  return c;
}
template <>
__host__ Vec3 h_random<HITTABLE>(const SceneObjects &s,
                                 const Point3 &o,
                                 int prim_idx) {
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype_);
  Vec3 c(0.0f);
  if (htype == SPHERE) {
    c = h_random<SPHERE>(s, o, prim_idx);
  } else if (htype == MOVING_SPHERE) {
    c = h_random<MOVING_SPHERE>(s, o, prim_idx);
  } else if (htype == XY_RECT) {
    c = h_random<XY_RECT>(s, o, prim_idx);
  } else if (htype == XZ_RECT) {
    c = h_random<XZ_RECT>(s, o, prim_idx);
  } else if (htype == YZ_RECT) {
    c = h_random<YZ_RECT>(s, o, prim_idx);
  } else if (htype == YZ_TRIANGLE) {
    c = h_random<YZ_TRIANGLE>(s, o, prim_idx);
  } else if (htype == XZ_TRIANGLE) {
    c = h_random<XZ_TRIANGLE>(s, o, prim_idx);
  } else if (htype == XY_TRIANGLE) {
    c = h_random<XY_TRIANGLE>(s, o, prim_idx);
  }
  return c;
}

template <GroupType g>
__device__ Vec3 random(const SceneObjects &s,
                       const Point3 &o, int group_idx,
                       curandState *loc) {
  return random_vec(loc);
}
template <GroupType g>
__host__ Vec3 h_random(const SceneObjects &s,
                       const Point3 &o, int group_idx) {
  return h_random_vec();
}
template <>
__device__ Vec3 random<NONE_GRP>(const SceneObjects &s,
                                 const Point3 &o,
                                 int group_idx,
                                 curandState *loc) {
  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  int obj_index =
      random_int(loc, group_start, group_size - 1);
  return random<HITTABLE>(s, o, obj_index, loc);
}
template <>
__host__ Vec3 h_random<NONE_GRP>(const SceneObjects &s,
                                 const Point3 &o,
                                 int group_idx) {
  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  int obj_index = h_random_int(group_start, group_size - 1);
  return h_random<HITTABLE>(s, o, obj_index);
}
template <>
__device__ Vec3 random<BOX>(const SceneObjects &s,
                            const Point3 &o, int group_idx,
                            curandState *loc) {
  return random<NONE_GRP>(s, o, group_idx, loc);
}
template <>
__device__ Vec3 random<CONSTANT_MEDIUM>(
    const SceneObjects &s, const Point3 &o, int group_idx,
    curandState *loc) {
  return random<NONE_GRP>(s, o, group_idx, loc);
}
template <>
__device__ Vec3 random<SIMPLE_MESH>(const SceneObjects &s,
                                    const Point3 &o,
                                    int group_idx,
                                    curandState *loc) {
  return random<NONE_GRP>(s, o, group_idx, loc);
}

template <>
__device__ Vec3 random<OBJECT>(const SceneObjects &s,
                               const Point3 &p,
                               int group_idx,
                               curandState *loc) {
  int gtype_ = s.gtypes[group_idx];
  GroupType gtype = static_cast<GroupType>(gtype_);
  Vec3 v;
  if (gtype == SIMPLE_MESH) {
    v = random<SIMPLE_MESH>(s, p, group_idx, loc);
  } else if (gtype == NONE_GRP) {
    v = random<NONE_GRP>(s, p, group_idx, loc);
  } else if (gtype == BOX) {
    v = random<BOX>(s, p, group_idx, loc);
  } else if (gtype == CONSTANT_MEDIUM) {
    v = random<CONSTANT_MEDIUM>(s, p, group_idx, loc);
  }
  return v;
}
//
template <>
__host__ Vec3 h_random<BOX>(const SceneObjects &s,
                            const Point3 &o,
                            int group_idx) {
  return h_random<NONE_GRP>(s, o, group_idx);
}
template <>
__host__ Vec3 h_random<CONSTANT_MEDIUM>(
    const SceneObjects &s, const Point3 &o, int group_idx) {
  return h_random<NONE_GRP>(s, o, group_idx);
}
template <>
__host__ Vec3 h_random<SIMPLE_MESH>(const SceneObjects &s,
                                    const Point3 &o,
                                    int group_idx) {
  return h_random<NONE_GRP>(s, o, group_idx);
}
template <>
__host__ Vec3 h_random<OBJECT>(const SceneObjects &s,
                               const Point3 &p,
                               int group_idx) {
  int gtype_ = s.gtypes[group_idx];
  GroupType gtype = static_cast<GroupType>(gtype_);
  Vec3 v;
  if (gtype == SIMPLE_MESH) {
    v = h_random<SIMPLE_MESH>(s, p, group_idx);
  } else if (gtype == NONE_GRP) {
    v = h_random<NONE_GRP>(s, p, group_idx);
  } else if (gtype == BOX) {
    v = h_random<BOX>(s, p, group_idx);
  } else if (gtype == CONSTANT_MEDIUM) {
    v = h_random<CONSTANT_MEDIUM>(s, p, group_idx);
  }
  return v;
}
