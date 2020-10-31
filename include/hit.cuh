#pragma once
// shade utils
#include <attenuation.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>
__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1.0f - (phi + M_PI) / (2.0f * M_PI);
  v = (theta + M_PI / 2.0f) / M_PI;
}

template <HittableType h>
__host__ __device__ bool hit(const SceneObjects &sobjs,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
  bool any_hit = false;
  return any_hit;
}

template <>
__host__ __device__ bool
hit<SPHERE>(const SceneObjects &s, const Ray &r,
            float d_min, float d_max, HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Point3 center(s.p1xs[prim_idx], s.p1ys[prim_idx],
                s.p1zs[prim_idx]);
  float radius = s.rads[prim_idx];
  Vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < d_max && temp > d_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      Vec3 normal = (rec.p - center) / radius;
      normal = to_unit(normal);
      rec.set_front_face(r, normal);
      get_sphere_uv(normal, rec.u, rec.v);
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < d_max && temp > d_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      Vec3 normal = (rec.p - center) / radius;
      normal = to_unit(normal);
      rec.set_front_face(r, normal);
      get_sphere_uv(normal, rec.u, rec.v);
      return true;
    }
  }
  return false;
}

template <>
__host__ __device__ bool
hit<MOVING_SPHERE>(const SceneObjects &s, const Ray &r,
                   float d_min, float d_max,
                   HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Point3 center1(s.p1xs[prim_idx], s.p1ys[prim_idx],
                 s.p1zs[prim_idx]);
  Point3 center2(s.p2xs[prim_idx], s.p2ys[prim_idx],
                 s.p2zs[prim_idx]);

  float radius = s.rads[prim_idx];
  float time0 = s.n1xs[prim_idx];
  float time1 = s.n1ys[prim_idx];

  float rt = r.time();
  Point3 scenter =
      moving_center(center1, center2, time0, time1, rt);
  Vec3 oc = r.origin() - scenter;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < d_max && temp > d_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      Vec3 normal = (rec.p - scenter) / radius;
      normal = to_unit(normal);
      rec.set_front_face(r, normal);
      get_sphere_uv(normal, rec.u, rec.v);
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < d_max && temp > d_min) {
      rec.t = temp;
      rec.p = r.at(rec.t);
      Vec3 normal = (rec.p - scenter) / radius;
      normal = to_unit(normal);
      rec.set_front_face(r, normal);
      get_sphere_uv(normal, rec.u, rec.v);

      return true;
    }
  }
  return false;
}
template <>
__host__ __device__ bool
hit<TRIANGLE>(const SceneObjects &s, const Ray &r,
              float d_min, float d_max, HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  Point3 p1(s.p1xs[prim_idx], s.p1ys[prim_idx],
            s.p1zs[prim_idx]);
  Point3 p2(s.p2xs[prim_idx], s.p2ys[prim_idx],
            s.p2zs[prim_idx]);
  Point3 p3(s.n1xs[prim_idx], s.n1ys[prim_idx],
            s.n1zs[prim_idx]);
  const float eps = 0.000001f;
  Vec3 edge1 = p1 - p2;
  Vec3 edge2 = p3 - p2;
  Vec3 h = cross(r.direction(), edge2);
  float a = dot(edge1, h);
  if (a > -eps && a < eps)
    return false; // ray parallel to triangle
  float f = 1.0f / a;
  Vec3 rToP2 = r.origin() - p2;
  float u = f * dot(rToP2, h);
  if (u < 0.0f || u > 1.0f)
    return false;

  Vec3 q = cross(rToP2, edge1);
  float v = f * dot(edge2, q);
  if (v < 0.0f || u + v > 1.0f)
    return false;

  float t = f * dot(r.direction(), q);
  if (t < eps)
    return false;

  rec.v = v;
  rec.u = u;
  rec.t = t;
  rec.p = r.at(rec.t);
  Vec3 outnormal = cross(edge1, edge2);
  rec.set_front_face(r, outnormal);
  return true;
}
template <>
__host__ __device__ bool
hit<XY_TRIANGLE>(const SceneObjects &s, const Ray &r,
                 float d_min, float d_max, HitRecord &rec) {
  return hit<TRIANGLE>(s, r, d_min, d_max, rec);
}
template <>
__host__ __device__ bool
hit<XZ_TRIANGLE>(const SceneObjects &s, const Ray &r,
                 float d_min, float d_max, HitRecord &rec) {
  return hit<TRIANGLE>(s, r, d_min, d_max, rec);
}
template <>
__host__ __device__ bool
hit<YZ_TRIANGLE>(const SceneObjects &s, const Ray &r,
                 float d_min, float d_max, HitRecord &rec) {
  return hit<TRIANGLE>(s, r, d_min, d_max, rec);
}
template <>
__host__ __device__ bool
hit<RECTANGLE>(const SceneObjects &s, const Ray &r,
               float d_min, float d_max, HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  HittableType htype =
      static_cast<HittableType>(s.htypes[prim_idx]);
  int aligned1, aligned2, not_aligned;
  if (htype == XY_RECT) {
    aligned1 = 0;
    aligned2 = 1;
    not_aligned = 2;
  } else if (htype == XZ_RECT) {
    aligned1 = 0;
    aligned2 = 2;
    not_aligned = 1;
  } else if (htype == YZ_RECT) {
    aligned1 = 1;
    aligned2 = 2;
    not_aligned = 0;
  }
  float k = s.p1zs[prim_idx];
  float a0 = s.p1xs[prim_idx];
  float b0 = s.p1ys[prim_idx];
  float a1 = s.p2xs[prim_idx];
  float b1 = s.p2ys[prim_idx];
  Vec3 anormal =
      to_unit(Vec3(s.n1xs[prim_idx], s.n1ys[prim_idx],
                   s.n1zs[prim_idx]));
  Point3 p1, p2, p3;
  p1[aligned1] = a0;
  p1[aligned2] = b0;
  p1[not_aligned] = k;
  p2[aligned1] = a1;
  p2[aligned2] = b0;
  p2[not_aligned] = k;
  p3[aligned1] = a1;
  p3[aligned2] = b1;
  p3[not_aligned] = k;
  anormal = to_unit(cross(p2 - p1, p3 - p1));

  float t = (k - r.origin()[not_aligned]) /
            r.direction()[not_aligned];
  if (t < d_min || t > d_max)
    return false;
  float a =
      r.origin()[aligned1] + t * r.direction()[aligned1];
  float b =
      r.origin()[aligned2] + t * r.direction()[aligned2];

  bool c1 = a0 < a and a < a1;
  bool c2 = b0 < b and b < b1;
  if ((c1 and c2) == false) {
    return false;
  }

  rec.u = (a - a0) / (a1 - a0);
  rec.v = (b - b0) / (b1 - b0);
  rec.t = t;
  rec.set_front_face(r, anormal);
  rec.p = r.at(t);
  return true;
}
template <>
__host__ __device__ bool
hit<XY_RECT>(const SceneObjects &s, const Ray &r,
             float d_min, float d_max, HitRecord &rec) {
  return hit<RECTANGLE>(s, r, d_min, d_max, rec);
}
template <>
__host__ __device__ bool
hit<XZ_RECT>(const SceneObjects &s, const Ray &r,
             float d_min, float d_max, HitRecord &rec) {
  return hit<RECTANGLE>(s, r, d_min, d_max, rec);
}
template <>
__host__ __device__ bool
hit<YZ_RECT>(const SceneObjects &s, const Ray &r,
             float d_min, float d_max, HitRecord &rec) {
  return hit<RECTANGLE>(s, r, d_min, d_max, rec);
}

template <>
__host__ __device__ bool
hit<HITTABLE>(const SceneObjects &s, const Ray &r,
              float d_min, float d_max, HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  int htype_ = s.htypes[prim_idx];
  HittableType htype = static_cast<HittableType>(htype_);
  bool res = false;
  if (htype == SPHERE) {
    res = hit<SPHERE>(s, r, d_min, d_max, rec);
  } else if (htype == MOVING_SPHERE) {
    res = hit<MOVING_SPHERE>(s, r, d_min, d_max, rec);
  } else if (htype == XY_RECT) {
    res = hit<XY_RECT>(s, r, d_min, d_max, rec);
  } else if (htype == XZ_RECT) {
    res = hit<XZ_RECT>(s, r, d_min, d_max, rec);
  } else if (htype == YZ_RECT) {
    res = hit<YZ_RECT>(s, r, d_min, d_max, rec);
  } else if (htype == TRIANGLE) {
    res = hit<TRIANGLE>(s, r, d_min, d_max, rec);
  } else if (htype == YZ_TRIANGLE) {
    res = hit<YZ_TRIANGLE>(s, r, d_min, d_max, rec);
  } else if (htype == XZ_TRIANGLE) {
    res = hit<XZ_TRIANGLE>(s, r, d_min, d_max, rec);
  } else if (htype == XY_TRIANGLE) {
    res = hit<XY_TRIANGLE>(s, r, d_min, d_max, rec);
  }
  return res;
}

template <GroupType g>
__device__ bool hit(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec, curandState *loc) {
  return false;
}
template <GroupType g>
__host__ bool h_hit(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec) {
  return false;
}
__host__ __device__ bool
hit_group_bbox(const SceneObjects &s, const Ray &r,
               float d_min, float d_max,
               const HitRecord &rec) {
  int group_index = rec.group_index;
  Point3 minp(s.g_minxs[group_index],
              s.g_minys[group_index],
              s.g_minzs[group_index]);
  Point3 maxp(s.g_maxxs[group_index],
              s.g_maxys[group_index],
              s.g_maxzs[group_index]);
  Aabb box(minp, maxp);
  return box.hit(r, d_min, d_max);
}
__host__ __device__ bool hit_group(const SceneObjects &s,
                                   const Ray &r,
                                   float d_min, float d_max,
                                   HitRecord &rec) {
  int group_index = rec.group_index;
  int group_start = s.group_starts[group_index];
  int group_size = s.group_sizes[group_index];
  int group_end = group_start + group_size;
  bool res = false;
  int j = 0;
  float closest_so_far = d_max;
  for (int i = group_start; i < group_end; i++) {
    rec.primitive_index = i;
    bool is_hit =
        hit<HITTABLE>(s, r, d_min, closest_so_far, rec);
    if (is_hit) {
      res = is_hit;
      j = i;
      closest_so_far = rec.t;
    }
  }
  rec.primitive_index = j;
  return res;
}
template <>
__device__ bool hit<NONE_GRP>(const SceneObjects &s,
                              const Ray &r, float d_min,
                              float d_max, HitRecord &rec,
                              curandState *loc) {
  return hit_group(s, r, d_min, d_max, rec);
}
template <>
__host__ bool h_hit<NONE_GRP>(const SceneObjects &s,
                              const Ray &r, float d_min,
                              float d_max, HitRecord &rec) {
  return hit_group(s, r, d_min, d_max, rec);
}
template <>
__device__ bool
hit<BOX>(const SceneObjects &s, const Ray &r, float d_min,
         float d_max, HitRecord &rec, curandState *loc) {
  return hit<NONE_GRP>(s, r, d_min, d_max, rec, loc);
}
template <>
__host__ bool h_hit<BOX>(const SceneObjects &s,
                         const Ray &r, float d_min,
                         float d_max, HitRecord &rec) {
  return h_hit<NONE_GRP>(s, r, d_min, d_max, rec);
}
__host__ __device__ bool
hit_constant_medium(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec, float t0, float t1) {
  const bool enableDebug = false;
  const bool debugging = enableDebug && t0 < 0.00001;

  HitRecord rec1, rec2;
  rec1.group_index = rec.group_index;
  rec2.group_index = rec.group_index;

  bool any_hit = hit_group(s, r, -FLT_MAX, FLT_MAX, rec1);
  if (!any_hit) {
    return any_hit;
  }

  any_hit = hit_group(s, r, rec1.t + 0.001, FLT_MAX, rec2);
  if (!any_hit) {
    return any_hit;
  }

  if (debugging) {
    printf("\nt0= %f", rec1.t);
    printf(", t1= %f\n", rec2.t);
  }

  if (rec1.t < d_min)
    rec1.t = d_min;
  if (rec2.t > d_max)
    rec2.t = d_max;

  if (rec1.t >= rec2.t)
    return false;

  if (rec1.t < 0)
    rec1.t = 0;

  const float neg_inv_dens =
      -1.0f / s.g_densities[rec.group_index];
  const float ray_length = r.direction().length();
  const float distance_inside_boundary =
      (rec2.t - rec1.t) * ray_length;
  const float hit_distance = neg_inv_dens * log(t1);

  if (hit_distance > distance_inside_boundary) {
    return false;
  }

  rec.t = rec1.t + hit_distance / ray_length;
  rec.p = r.at(rec.t);

  if (debugging) {
    printf("hit_distance = %f\n", hit_distance);
    printf("rec.t = %f\n", rec.t);
    printf("rec.p = %f ", rec.p.x());
    printf("%f ", rec.p.y());
    printf("%f ", rec.p.z());
  }

  rec.normal = Vec3(1, 0, 0); // arbitrary
  rec.front_face = true;      // also arbitrary
  rec.group_scattering = true;

  return true;
}
template <>
__device__ bool
hit<CONSTANT_MEDIUM>(const SceneObjects &s, const Ray &r,
                     float d_min, float d_max,
                     HitRecord &rec, curandState *loc) {
  // Print occasional samples when debugging. To enable,
  float t0 = curand_uniform(loc);
  float t1 = curand_uniform(loc);
  return hit_constant_medium(s, r, d_min, d_max, rec, t0,
                             t1);
}
template <>
__host__ bool
h_hit<CONSTANT_MEDIUM>(const SceneObjects &s, const Ray &r,
                       float d_min, float d_max,
                       HitRecord &rec) {
  // Print occasional samples when debugging. To enable,
  float t0 = hrandf();
  float t1 = hrandf();
  return hit_constant_medium(s, r, d_min, d_max, rec, t0,
                             t1);
}
template <>
__device__ bool
hit<SIMPLE_MESH>(const SceneObjects &s, const Ray &r,
                 float d_min, float d_max, HitRecord &rec,
                 curandState *loc) {
  if (!hit_group_bbox(s, r, d_min, d_max, rec)) {
    return false;
  }
  return hit<NONE_GRP>(s, r, d_min, d_max, rec, loc);
}
template <>
__host__ bool h_hit<SIMPLE_MESH>(const SceneObjects &s,
                                 const Ray &r, float d_min,
                                 float d_max,
                                 HitRecord &rec) {
  if (!hit_group_bbox(s, r, d_min, d_max, rec)) {
    return false;
  }

  return h_hit<NONE_GRP>(s, r, d_min, d_max, rec);
}
template <>
__device__ bool hit<OBJECT>(const SceneObjects &s,
                            const Ray &r, float d_min,
                            float d_max, HitRecord &rec,
                            curandState *loc) {
  int gtype_ = s.gtypes[rec.group_index];
  GroupType gtype = static_cast<GroupType>(gtype_);
  bool is_hit = false;
  if (gtype == NONE_GRP) {
    is_hit = hit<NONE_GRP>(s, r, d_min, d_max, rec, loc);
  } else if (gtype == BOX) {
    is_hit = hit<BOX>(s, r, d_min, d_max, rec, loc);
  } else if (gtype == CONSTANT_MEDIUM) {
    is_hit =
        hit<CONSTANT_MEDIUM>(s, r, d_min, d_max, rec, loc);
  } else if (gtype == SIMPLE_MESH) {
    is_hit = hit<SIMPLE_MESH>(s, r, d_min, d_max, rec, loc);
  }
  return is_hit;
}

template <>
__host__ bool h_hit<OBJECT>(const SceneObjects &s,
                            const Ray &r, float d_min,
                            float d_max, HitRecord &rec) {
  int gtype_ = s.gtypes[rec.group_index];
  GroupType gtype = static_cast<GroupType>(gtype_);
  bool is_hit = false;
  if (gtype == NONE_GRP) {
    is_hit = h_hit<NONE_GRP>(s, r, d_min, d_max, rec);
  } else if (gtype == BOX) {
    is_hit = h_hit<BOX>(s, r, d_min, d_max, rec);
  } else if (gtype == CONSTANT_MEDIUM) {
    is_hit =
        h_hit<CONSTANT_MEDIUM>(s, r, d_min, d_max, rec);
  } else if (gtype == SIMPLE_MESH) {
    is_hit = h_hit<SIMPLE_MESH>(s, r, d_min, d_max, rec);
  }
  return is_hit;
}

template <TransformationType T>
__host__ bool h_hit(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec) {
  return false;
}
template <TransformationType T>
__device__ bool hit(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec, curandState *loc) {
  return false;
}
template <>
__host__ bool
h_hit<NONE_TRANSFORMATION>(const SceneObjects &s,
                           const Ray &r, float d_min,
                           float d_max, HitRecord &rec) {
  return h_hit<OBJECT>(s, r, d_min, d_max, rec);
}
template <>
__device__ bool hit<NONE_TRANSFORMATION>(
    const SceneObjects &s, const Ray &r, float d_min,
    float d_max, HitRecord &rec, curandState *loc) {
  return hit<OBJECT>(s, r, d_min, d_max, rec, loc);
}
template <>
__host__ bool
h_hit<TRANSLATE>(const SceneObjects &s, const Ray &r,
                 float d_min, float d_max, HitRecord &rec) {
  int g_index = rec.group_index;
  Point3 disp(s.g_stepxs[g_index], s.g_stepys[g_index],
              s.g_stepzs[g_index]);
  Ray nr(r.origin() - disp, r.direction(), r.time());
  if (!h_hit<OBJECT>(s, nr, d_min, d_max, rec))
    return false;

  rec.p += disp;
  rec.set_front_face(nr, rec.normal);
  return true;
}
template <>
__device__ bool hit<TRANSLATE>(const SceneObjects &s,
                               const Ray &r, float d_min,
                               float d_max, HitRecord &rec,
                               curandState *loc) {
  int g_index = rec.group_index;
  Point3 disp(s.g_stepxs[g_index], s.g_stepys[g_index],
              s.g_stepzs[g_index]);
  Ray nr(r.origin() - disp, r.direction(), r.time());
  if (!hit<OBJECT>(s, nr, d_min, d_max, rec, loc))
    return false;

  rec.p += disp;
  rec.set_front_face(nr, rec.normal);
  return true;
}

template <>
__host__ bool h_hit<ROTATE>(const SceneObjects &s,
                            const Ray &r, float d_min,
                            float d_max, HitRecord &rec) {
  int g_index = rec.group_index;
  float deg = s.g_degrees[g_index];
  float rad = degree_to_radian(deg);
  Vec3 axis =
      Vec3(s.g_axis_xs[g_index], s.g_axis_ys[g_index],
           s.g_axis_zs[g_index]);
  Matrix rotMat = rotate(axis, rad);
  Point3 neworig = rotMat * r.origin();
  Vec3 ndir = rotMat * r.direction();
  Ray rotated_r = Ray(neworig, ndir, r.time());

  if (!h_hit<OBJECT>(s, rotated_r, d_min, d_max, rec))
    return false;

  rotMat = rotate(axis, -rad);
  Point3 p = rec.p;
  Vec3 norm = rec.normal;
  Point3 np = rotMat * p;
  Vec3 nnorm = rotMat * norm;
  rec.p = np;
  rec.set_front_face(rotated_r, nnorm);
  return true;
}
template <>
__device__ bool hit<ROTATE>(const SceneObjects &s,
                            const Ray &r, float d_min,
                            float d_max, HitRecord &rec,
                            curandState *loc) {
  int g_index = rec.group_index;
  float deg = s.g_degrees[g_index];
  float rad = degree_to_radian(deg);
  Vec3 axis =
      Vec3(s.g_axis_xs[g_index], s.g_axis_ys[g_index],
           s.g_axis_zs[g_index]);
  Matrix rotMat = rotate(axis, rad);
  Point3 neworig = rotMat * r.origin();
  Vec3 ndir = rotMat * r.direction();
  Ray rotated_r = Ray(neworig, ndir, r.time());

  if (!hit<OBJECT>(s, rotated_r, d_min, d_max, rec, loc))
    return false;

  rotMat = rotate(axis, -rad);
  Point3 p = rec.p;
  Vec3 norm = rec.normal;
  Point3 np = rotMat * p;
  Vec3 nnorm = rotMat * norm;
  rec.p = np;
  rec.set_front_face(rotated_r, nnorm);
  return true;
}

template <>
__host__ bool
h_hit<ROTATE_TRANSLATE>(const SceneObjects &s, const Ray &r,
                        float d_min, float d_max,
                        HitRecord &rec) {
  bool res = h_hit<ROTATE>(s, r, d_min, d_max, rec);
  res = h_hit<TRANSLATE>(s, r, d_min, d_max, rec);
  return res;
}

template <>
__host__ bool
h_hit<TRANSFORMATION>(const SceneObjects &s, const Ray &r,
                      float d_min, float d_max,
                      HitRecord &rec) {
  TransformationType ttype =
      static_cast<TransformationType>(
          s.g_transtypes[rec.group_index]);
  bool res = false;
  if (ttype == TRANSLATE) {
    res = h_hit<TRANSLATE>(s, r, d_min, d_max, rec);
  } else if (ttype == ROTATE) {
    res = h_hit<ROTATE>(s, r, d_min, d_max, rec);
  } else if (ttype == ROTATE_TRANSLATE) {
    res = h_hit<ROTATE_TRANSLATE>(s, r, d_min, d_max, rec);
  } else {
    res =
        h_hit<NONE_TRANSFORMATION>(s, r, d_min, d_max, rec);
  }
  return res;
}

template <>
__device__ bool
hit<TRANSFORMATION>(const SceneObjects &s, const Ray &r,
                    float d_min, float d_max,
                    HitRecord &rec, curandState *loc) {
  TransformationType ttype =
      static_cast<TransformationType>(
          s.g_transtypes[rec.group_index]);
  bool res = false;
  if (ttype == TRANSLATE) {
    res = hit<TRANSLATE>(s, r, d_min, d_max, rec, loc);
  } else if (ttype == ROTATE) {
    res = hit<ROTATE>(s, r, d_min, d_max, rec, loc);
  } else {
    res = hit<NONE_TRANSFORMATION>(s, r, d_min, d_max, rec,
                                   loc);
  }
  return res;
}

template <>
__device__ bool
hit<SCENE>(const SceneObjects &s, const Ray &r, float d_min,
           float d_max, HitRecord &rec, curandState *loc) {
  int nb_group = s.nb_groups;
  bool res = false;
  int g_index = 0;
  int p_index = 0;
  float closest_so_far = d_max;
  for (int i = 0; i < nb_group; i++) {
    rec.group_index = i;
    bool is_hit = hit<TRANSFORMATION>(
        s, r, d_min, closest_so_far, rec, loc);
    if (is_hit) {
      res = is_hit;
      g_index = i;
      closest_so_far = rec.t;
      p_index = rec.primitive_index;
    }
  }

  bool group_scattering = false;
  int gtype_ = s.gtypes[g_index];
  GroupType gtype = static_cast<GroupType>(gtype_);
  if (gtype == CONSTANT_MEDIUM) {
    group_scattering = true;
  }
  rec.group_index = g_index;
  rec.primitive_index = p_index;
  rec.group_scattering = group_scattering;
  return res;
}

template <>
__host__ bool h_hit<SCENE>(const SceneObjects &s,
                           const Ray &r, float d_min,
                           float d_max, HitRecord &rec) {
  int nb_group = s.nb_groups;
  bool res = false;
  int g_index = 0;
  int p_index = 0;
  float closest_so_far = d_max;
  for (int i = 0; i < nb_group; i++) {
    rec.group_index = i;
    bool is_hit = h_hit<TRANSFORMATION>(
        s, r, d_min, closest_so_far, rec);
    if (is_hit) {
      res = is_hit;
      g_index = i;
      closest_so_far = rec.t;
      p_index = rec.primitive_index;
    }
  }

  bool group_scattering = false;
  int gtype_ = s.gtypes[g_index];
  GroupType gtype = static_cast<GroupType>(gtype_);
  if (gtype == CONSTANT_MEDIUM) {
    group_scattering = true;
  }
  rec.group_index = g_index;
  rec.primitive_index = p_index;
  rec.group_scattering = group_scattering;
  return res;
}
