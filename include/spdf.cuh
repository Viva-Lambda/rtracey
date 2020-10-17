#pragma once
// surface pdf value
#include <aabb.cuh>
#include <hit.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <sceneshape.cuh>
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
pdf_value<SPHERE_HIT>(const SceneObjects &s,
                      const Point3 &o, const Point3 &v,
                      int prim_idx) {
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<SPHERE_HIT>(s, Ray(o, v), 0.001, FLT_MAX, rec))
    return 0.0f;

  float rad2 = radius * radius;
  Vec3 cent_diff = center - o;
  auto cos_theta_max =
      sqrt(1 - rad2 / cent_diff.squared_length());
  auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

  return 1 / solid_angle;
}

template <>
__host__ __device__ float pdf_value<MOVING_SPHERE_HIT>(
    const SceneObjects &s, const Point3 &o, const Point3 &v,
    int prim_idx) {
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  float radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];
  Point3 center = MovingSphere::mcenter(center1, center2,
                                        time0, time1, rt);

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<MOVING_SPHERE_HIT>(s, Ray(o, v), 0.001, FLT_MAX,
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
pdf_value<TRIANGLE_HIT>(const SceneObjects &s,
                        const Point3 &o, const Point3 &v,
                        int prim_idx) {
  return 1.0f;
}

template <>
__host__ __device__ float
pdf_value<RECT_HIT>(const SceneObjects &s, const Point3 &o,
                    const Point3 &v, int prim_idx) {
  float k = s.rads[prim_idx];
  float a0 = s.p1xs[prim_idx];
  float a1 = s.p1ys[prim_idx];
  float b0 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Vec3 anormal = Vec3(s.n1xs[prim_idx], s.n1ys[prim_idx],
                      s.n1zs[prim_idx]);
  AxisInfo ax = AxisInfo(anormal);

  HitRecord rec;
  rec.primitive_index = prim_idx;
  if (!hit<RECT_HIT>(rect, Ray(o, v), 0.001, FLT_MAX, rec))
    return 0;

  float area = (a1 - a0) * (b1 - b0);
  return get_pdf_surface(v, rec.normal, rec.t, area);
}

template <>
__host__ __device__ float
pdf_value<XY_RECT_HIT>(const SceneObjects &s,
                       const Point3 &o, const Point3 &v,
                       int prim_idx) {
  return pdf_value<RECT_HIT>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<XZ_RECT_HIT>(const SceneObjects &s,
                       const Point3 &o, const Point3 &v,
                       int prim_idx) {
  return pdf_value<RECT_HIT>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<YZ_RECT_HIT>(const SceneObjects &s,
                       const Point3 &o, const Point3 &v,
                       int prim_idx) {
  return pdf_value<RECT_HIT>(s, o, v, prim_idx);
}

template <>
__host__ __device__ float
pdf_value<HITTABLE>(const SceneObjects &s, const Point3 &o,
                    const Point3 &v, int prim_idx) {
  //
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype);
  float pdf = 0.0f;
  if (htype == SPHERE_HIT) {
    pdf = pdf_value<SPHERE_HIT>(s, o, v, prim_idx);
  } else if (htype == MOVING_SPHERE_HIT) {
    pdf = pdf_value<MOVING_SPHERE_HIT>(s, o, v, prim_idx);
  } else if (XY_TRIANGLE_HIT) {
    pdf = pdf_value<XY_TRIANGLE_HIT>(s, o, v, prim_idx);
  } else if (XZ_TRIANGLE_HIT) {
    pdf = pdf_value<XZ_TRIANGLE_HIT>(s, o, v, prim_idx);
  } else if (YZ_TRIANGLE_HIT) {
    pdf = pdf_value<YZ_TRIANGLE_HIT>(s, o, v, prim_idx);
  } else if (TRIANGLE_HIT) {
    pdf = pdf_value<TRIANGLE_HIT>(s, o, v, prim_idx);
  } else if (RECT_HIT) {
    pdf = pdf_value<RECT_HIT>(s, o, v, prim_idx);
  } else if (XY_RECT_HIT) {
    pdf = pdf_value<XY_RECT_HIT>(s, o, v, prim_idx);
  } else if (XZ_RECT_HIT) {
    pdf = pdf_value<XZ_RECT_HIT>(s, o, v, prim_idx);
  } else if (YZ_RECT_HIT) {
    pdf = pdf_value<YZ_RECT_HIT>(s, o, v, prim_idx);
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
    sum += weight * pdf_value<HITTABLE>(p, o, v);
  }
  return sum;
}

template <>
__host__ __device__ float
pdf_value<BOX_GRP>(const SceneObjects &s, const Point3 &o,
                   const Point3 &v, int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}
template <>
__host__ __device__ float pdf_value<CONSTANT_MEDIUM_GRP>(
    const SceneObjects &s, const Point3 &o, const Point3 &v,
    int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}
template <>
__host__ __device__ float
pdf_value<SIMPLE_MESH_GRP>(const SceneObjects &s,
                           const Point3 &o, const Point3 &v,
                           int group_idx) {
  return pdf_value<NONE_GRP>(s, o, v, group_idx);
}
template <>
__host__ __device__ float
pdf_value<SCENE_GRP>(const SceneObjects &s, const Point3 &o,
                     const Point3 &v, int group_idx = 0) {

  int nb_group = s.nb_groups;
  float pdf_v = 1.0f;
  for (int i = 0; i < nb_group; i++) {
    int group_idx = i;
    int gtype_ = s.g_ttypes[group_idx];
    GroupType gtype = static_cast<GroupType>(gtype_);
    if (gtype == SIMPLE_MESH_GRP) {
      pdf_v =
          pdf_value<SIMPLE_MESH_GRP>(s, o, v, group_idx);
      return pdf_v;
    } else if (gtype == NONE_GRP) {
      pdf_v = pdf_value<NONE_GRP>(s, o, v, group_idx);
      return pdf_v;
    } else if (gtype == BOX_GRP) {
      pdf_v = pdf_value<BOX_GRP>(s, o, v, group_idx);
      return pdf_v;
    } else if (gtype == CONSTANT_MEDIUM_GRP) {
      pdf_v = pdf_value<CONSTANT_MEDIUM_GRP>(s, o, v,
                                             group_idx);
      return pdf_v;
    }
  }
  return pdf_v;
}
