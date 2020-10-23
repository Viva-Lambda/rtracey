#pragma once

#include <onb.cuh>
#include <prandom.cuh>
#include <ray.cuh>
#include <spdf.cuh>
#include <vec3.cuh>

template <PdfType p>
__host__ __device__ float
pdf_value(const SceneObjects &s, const HitRecord &rec,
          const ScatterRecord &srec) {
  return 0.0f;
}
template <>
__host__ __device__ float
pdf_value<COSINE_PDF>(const SceneObjects &s,
                      const HitRecord &rec,
                      const ScatterRecord &srec) {
  return dot(rec.normal, srec.specular_ray.direction()) /
         M_PI;
}
template <>
__host__ __device__ float
pdf_value<HITTABLE_PDF>(const SceneObjects &s,
                        const HitRecord &rec,
                        const ScatterRecord &srec) {
  Point3 origin = rec.p;
  Point3 direction = srec.specular_ray.direction();
  int prim_idx;
  float pdf;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    pdf = pdf_value<OBJECT>(s, origin, direction, prim_idx);
  } else {
    prim_idx = rec.primitive_index;
    pdf =
        pdf_value<HITTABLE>(s, origin, direction, prim_idx);
  }
  return pdf;
}
template <>
__host__ __device__ float
pdf_value<MIXTURE_PDF>(const SceneObjects &s,
                       const HitRecord &rec,
                       const ScatterRecord &srec) {
  Point3 origin = rec.p;
  Point3 direction = srec.specular_ray.direction();

  float weight = 1.0f / srec.index_size;
  float sum = 0.0f;
  for (int i = 0; i < srec.index_size; i++) {
    //
    bool is_group_index = srec.is_group_indices[i];
    int index = srec.indices[i];
    float pdf;
    if (is_group_index) {
      pdf = pdf_value<OBJECT>(s, origin, direction, index);
    } else {
      pdf =
          pdf_value<HITTABLE>(s, origin, direction, index);
    }
    sum += pdf;
  }
  return sum;
}
template <PdfType p>
__device__ Vec3 pdf_generate(const SceneObjects &s,
                             const HitRecord &rec,
                             const ScatterRecord &srec,
                             curandState *loc) {
  return Vec3(0.0f);
}
template <PdfType p>
__host__ Vec3 h_pdf_generate(const SceneObjects &s,
                             const HitRecord &rec,
                             const ScatterRecord &srec) {
  return Vec3(0.0f);
}
template <>
__device__ Vec3 pdf_generate<COSINE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec, curandState *loc) {
  Onb uvw;
  uvw.build_from_w(rec.normal);
  return uvw.local(random_cosine_direction(loc));
}
template <>
__host__ Vec3 h_pdf_generate<COSINE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec) {
  Onb uvw;
  uvw.build_from_w(rec.normal);
  return uvw.local(h_random_cosine_direction());
}
template <>
__device__ Vec3 pdf_generate<HITTABLE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec, curandState *loc) {
  Point3 origin = rec.p;
  int idx;
  Vec3 dir;
  if (rec.group_scattering) {
    idx = rec.group_index;
    dir = random<OBJECT>(s, origin, idx, loc);
  } else {
    idx = rec.primitive_index;
    dir = random<HITTABLE>(s, origin, idx, loc);
  }
  return dir;
}
template <>
__host__ Vec3 h_pdf_generate<HITTABLE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec) {
  Point3 origin = rec.p;
  int idx;
  Vec3 dir;
  if (rec.group_scattering) {
    idx = rec.group_index;
    dir = h_random<OBJECT>(s, origin, idx);
  } else {
    idx = rec.primitive_index;
    dir = h_random<HITTABLE>(s, origin, idx);
  }
  return dir;
}
template <>
__device__ Vec3 pdf_generate<MIXTURE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec, curandState *loc) {
  Point3 origin = rec.p;
  int obj_index = random_int(loc, 0, srec.index_size);
  bool is_group_index = srec.is_group_indices[obj_index];
  int index;
  Vec3 dir;
  if (is_group_index) {
    index = srec.indices[obj_index];
    dir = random<OBJECT>(s, origin, index, loc);
  } else {
    index = srec.indices[obj_index];
    dir = random<HITTABLE>(s, origin, index, loc);
  }
  return dir;
}
template <>
__host__ Vec3 h_pdf_generate<MIXTURE_PDF>(
    const SceneObjects &s, const HitRecord &rec,
    const ScatterRecord &srec) {
  Point3 origin = rec.p;
  int obj_index = h_random_int(0, srec.index_size);
  bool is_group_index = srec.is_group_indices[obj_index];
  int index;
  Vec3 dir;
  if (is_group_index) {
    index = srec.indices[obj_index];
    dir = h_random<OBJECT>(s, origin, index);
  } else {
    index = srec.indices[obj_index];
    dir = h_random<HITTABLE>(s, origin, index);
  }
  return dir;
}
