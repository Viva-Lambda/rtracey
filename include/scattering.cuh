#pragma once
// scattering related functions
#include <attenuation.cuh>
#include <onb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <MaterialType m>
__device__ bool scatter(const SceneObjects &s, const Ray &r,
                        const HitRecord &rec,
                        Color &attenuation, Ray &r_out,
                        float &pdf, curandState *loc) {
  return false;
}
template <>
__device__ bool
scatter<LAMBERTIAN>(const SceneObjects &s, const Ray &r,
                    const HitRecord &rec,
                    Color &attenuation, Ray &r_out,
                    float &pdf, curandState *loc) {
  Onb uvw;
  uvw.build_from_w(rec.normal);
  auto direction = uvw.local(random_cosine_direction(loc));
  r_out = Ray(rec.p, to_unit(direction), r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  // attenuation = Color(0.0f, 1.0f, 0.0f);
  pdf = dot(uvw.w(), r_out.direction()) / M_PI;
  return true;
}
template <>
__device__ bool
scatter<METAL>(const SceneObjects &s, const Ray &r,
               const HitRecord &rec, Color &attenuation,
               Ray &r_out, float &pdf, curandState *loc) {
  int prim_idx = rec.primitive_index;
  float fuzz = s.fuzz_ref_idxs[prim_idx];
  Vec3 reflected =
      reflect(to_unit(r.direction()), rec.normal);
  r_out = Ray(rec.p,
              reflected + fuzz * random_in_unit_sphere(loc),
              r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  pdf = 1.0f;
  return (dot(r_out.direction(), rec.normal) > 0.0f);
}
template <>
__device__ bool
scatter<DIELECTRIC>(const SceneObjects &s, const Ray &r,
                    const HitRecord &rec,
                    Color &attenuation, Ray &r_out,
                    float &pdf, curandState *loc) {
  int prim_idx = rec.primitive_index;
  float ref_idx = s.fuzz_ref_idxs[prim_idx];
  pdf = 1.0f;
  Vec3 outward_normal;
  Vec3 reflected = reflect(r.direction(), rec.normal);
  float ni_over_nt;
  attenuation = Vec3(1.0f, 1.0f, 1.0f);
  Vec3 refracted;
  float reflect_prob;
  float cosine;
  if (dot(r.direction(), rec.normal) > 0.0f) {
    outward_normal = -rec.normal;
    ni_over_nt = ref_idx;
    cosine = dot(r.direction(), rec.normal) /
             r.direction().length();
    cosine = sqrt(1.0f -
                  ni_over_nt * ni_over_nt *
                      (1 - cosine * cosine));
  } else {
    outward_normal = rec.normal;
    ni_over_nt = 1.0f / ref_idx;
    cosine = -dot(r.direction(), rec.normal) /
             r.direction().length();
  }
  if (refract(r.direction(), outward_normal, ni_over_nt,
              refracted))
    reflect_prob = fresnelCT(cosine, ref_idx);
  else
    reflect_prob = 1.0f;
  if (curand_uniform(loc) < reflect_prob)
    r_out = Ray(rec.p, reflected, r.time());
  else
    r_out = Ray(rec.p, refracted, r.time());
  return true;
}
template <>
__device__ bool
scatter<DIFFUSE_LIGHT>(const SceneObjects &s, const Ray &r,
                       const HitRecord &rec,
                       Color &attenuation, Ray &r_out,
                       float &pdf, curandState *loc) {
  pdf = 1.0f;
  return false;
}
template <>
__device__ bool
scatter<ISOTROPIC>(const SceneObjects &s, const Ray &r,
                   const HitRecord &rec, Color &attenuation,
                   Ray &r_out, float &pdf,
                   curandState *loc) {
  r_out = Ray(rec.p, random_in_unit_sphere(loc), r.time());
  attenuation = color_value<TEXTURE>(s, rec);
  pdf = 1.0f;
  return true;
}
template <>
__device__ bool
scatter<MATERIAL>(const SceneObjects &s, const Ray &r,
                  const HitRecord &rec, Color &attenuation,
                  Ray &r_out, float &pdf,
                  curandState *loc) {
  int prim_idx = rec.primitive_index;
  bool res = false;
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
  if (mtype == LAMBERTIAN) {
    res = scatter<LAMBERTIAN>(s, r, rec, attenuation, r_out,
                              pdf, loc);
  } else if (mtype == METAL) {
    res = scatter<METAL>(s, r, rec, attenuation, r_out, pdf,
                         loc);
  } else if (mtype == DIELECTRIC) {
    res = scatter<DIELECTRIC>(s, r, rec, attenuation, r_out,
                              pdf, loc);
  } else if (mtype == DIFFUSE_LIGHT) {
    res = scatter<DIFFUSE_LIGHT>(s, r, rec, attenuation,
                                 r_out, pdf, loc);
  } else if (mtype == ISOTROPIC) {
    res = scatter<ISOTROPIC>(s, r, rec, attenuation, r_out,
                             pdf, loc);
  }
  // attenuation = Color(1.0f, 1.0f, 0.0f);
  return res;
}

__host__ __device__ MaterialType scatter_material(
    const SceneObjects &s, const HitRecord &rec) {
  int prim_idx = rec.primitive_index;
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
  return mtype;
}

template <MaterialType m>
__device__ float
scattering_pdf(const SceneObjects &s, const Ray &r_in,
               const HitRecord &rec, const Ray &r_out) {
  return 0.0f;
}

template <>
__device__ float scattering_pdf<LAMBERTIAN>(
    const SceneObjects &s, const Ray &r_in,
    const HitRecord &rec, const Ray &r_out) {
  auto cosine = dot(rec.normal, to_unit(r_out.direction()));
  return cosine < 0 ? 0 : cosine / M_PI;
}
template <>
__device__ float scattering_pdf<MATERIAL>(
    const SceneObjects &s, const Ray &r_in,
    const HitRecord &rec, const Ray &r_out) {
  int prim_idx = rec.primitive_index;
  float res = 0.0f;
  MaterialType mtype =
      static_cast<MaterialType>(s.mtypes[prim_idx]);
  if (mtype == LAMBERTIAN) {
    res = scattering_pdf<LAMBERTIAN>(s, r_in, rec, r_out);
  }
  return res;
}
