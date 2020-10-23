#pragma once
// surface pdf value
#include <aabb.cuh>
#include <hit.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>
__host__ __device__ float get_pdf_surface(Vec3 dir,
                                          Vec3 normal,
                                          float dist,
                                          float area) {
  float dist_squared = dist * dist / dir.squared_length();
  float cosine = fabs(dot(dir, normal) / dir.length());
  return dist_squared / (cosine * area);
}

template <HittableType h>
__host__ __device__ float
pdf_value(const SceneObjects &s, const Point3 &o,
          const Point3 &v, int prim_idx) {
  return 1.0f;
}

template <>
__host__ __device__ float
pdf_value<SPHERE>(const SceneObjects &s, const Point3 &o,
                  const Point3 &v, int prim_idx) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<SPHERE>(s, Ray(o, v), 0.001, FLT_MAX, rec))
    return 0.0f;

  float rad2 = radius * radius;
  Vec3 cent_diff = center - o;
  auto cos_theta_max =
      sqrt(1 - rad2 / cent_diff.squared_length());
  auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

  return 1 / solid_angle;
}

template <>
__host__ __device__ float
pdf_value<MOVING_SPHERE>(const SceneObjects &s,
                         const Point3 &o, const Point3 &v,
                         int prim_idx) {
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  float radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];
  float rt = time1 - time0;
  Point3 center =
      moving_center(center1, center2, time0, time1, rt);

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<MOVING_SPHERE>(s, Ray(o, v), 0.001, FLT_MAX,
                          rec))
    return 0.0f;

  float rad2 = radius * radius;
  Vec3 cent_diff = center - o;
  auto cos_theta_max =
      sqrt(1 - rad2 / cent_diff.squared_length());
  auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

  return 1 / solid_angle;
}

template <>
__host__ __device__ float
pdf_value<TRIANGLE>(const SceneObjects &s, const Point3 &o,
                    const Point3 &v, int prim_idx) {
  return 1.0f;
}

template <>
__host__ __device__ float
pdf_value<RECTANGLE>(const SceneObjects &s, const Point3 &o,
                     const Point3 &v, int prim_idx) {
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p1ys[prim_idx];
  float b0 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Vec3 anormal = Vec3(s.n1xs[prim_idx], s.n1ys[prim_idx],
                      s.n1zs[prim_idx]);
  AxisInfo ax = AxisInfo(s.rads[prim_idx]);

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<RECTANGLE>(s, Ray(o, v), 0.001, FLT_MAX, rec))
    return 0;

  float area = (a1 - a0) * (b1 - b0);
  return get_pdf_surface(v, rec.normal, rec.t, area);
}

template <>
__host__ __device__ float
pdf_value<XY_RECT>(const SceneObjects &s, const Point3 &o,
                   const Point3 &v, int prim_idx) {
  return pdf_value<RECTANGLE>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<XZ_RECT>(const SceneObjects &s, const Point3 &o,
                   const Point3 &v, int prim_idx) {
  return pdf_value<RECTANGLE>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<YZ_RECT>(const SceneObjects &s, const Point3 &o,
                   const Point3 &v, int prim_idx) {
  return pdf_value<RECTANGLE>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<HITTABLE>(const SceneObjects &s, const Point3 &o,
                    const Point3 &v, int prim_idx) {
  //
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype_);
  float pdf = 0.0f;
  if (htype == SPHERE) {
    pdf = pdf_value<SPHERE>(s, o, v, prim_idx);
  } else if (htype == MOVING_SPHERE) {
    pdf = pdf_value<MOVING_SPHERE>(s, o, v, prim_idx);
  } else if (XY_TRIANGLE) {
    pdf = pdf_value<XY_TRIANGLE>(s, o, v, prim_idx);
  } else if (XZ_TRIANGLE) {
    pdf = pdf_value<XZ_TRIANGLE>(s, o, v, prim_idx);
  } else if (YZ_TRIANGLE) {
    pdf = pdf_value<YZ_TRIANGLE>(s, o, v, prim_idx);
  } else if (TRIANGLE) {
    pdf = pdf_value<TRIANGLE>(s, o, v, prim_idx);
  } else if (RECTANGLE) {
    pdf = pdf_value<RECTANGLE>(s, o, v, prim_idx);
  } else if (XY_RECT) {
    pdf = pdf_value<XY_RECT>(s, o, v, prim_idx);
  } else if (XZ_RECT) {
    pdf = pdf_value<XZ_RECT>(s, o, v, prim_idx);
  } else if (YZ_RECT) {
    pdf = pdf_value<YZ_RECT>(s, o, v, prim_idx);
  }
  return pdf;
}

template <GroupType g>
__host__ __device__ float
pdf_value(const SceneObjects &s, const Point3 &o,
          const Point3 &v, int group_idx) {
  return 1.0f;
}
template <>
__host__ __device__ float
pdf_value<NONE_GRP>(const SceneObjects &s, const Point3 &o,
                    const Point3 &v, int group_idx) {
  int group_start = s.group_starts[group_idx];
  int group_size = s.group_sizes[group_idx];
  float weight = 1.0f / group_size;
  float sum = 0.0f;
  for (int i = group_start; i < group_size; i++) {
    int prim_idx = i;
    sum += weight * pdf_value<HITTABLE>(s, o, v, prim_idx);
  }
  return sum;
}

template <>
__host__ __device__ float
pdf_value<BOX>(const SceneObjects &s, const Point3 &o,
               const Point3 &v, int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}
template <>
__host__ __device__ float
pdf_value<CONSTANT_MEDIUM>(const SceneObjects &s,
                           const Point3 &o, const Point3 &v,
                           int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}
template <>
__host__ __device__ float
pdf_value<SIMPLE_MESH>(const SceneObjects &s,
                       const Point3 &o, const Point3 &v,
                       int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}

template <>
__host__ __device__ float
pdf_value<OBJECT>(const SceneObjects &s, const Point3 &o,
                  const Point3 &v, int group_idx) {
  int gtype_ = s.gtypes[group_idx];
  GroupType gtype = static_cast<GroupType>(gtype_);
  float pdf_v = 1.0f;
  if (gtype == SIMPLE_MESH) {
    pdf_v = pdf_value<SIMPLE_MESH>(s, o, v, group_idx);
  } else if (gtype == NONE_GRP) {
    pdf_v = pdf_value<NONE_GRP>(s, o, v, group_idx);
  } else if (gtype == BOX) {
    pdf_v = pdf_value<BOX>(s, o, v, group_idx);
  } else if (gtype == CONSTANT_MEDIUM) {
    pdf_v = pdf_value<CONSTANT_MEDIUM>(s, o, v, group_idx);
  }
  return pdf_v;
}
