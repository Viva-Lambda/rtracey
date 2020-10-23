#pragma once
// scattering related functions
#include <attenuation.cuh>
#include <onb.cuh>
#include <pdf.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneobj.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <MaterialType m>
__device__ bool scatter(const SceneObjects &s, const Ray &r,
                        const HitRecord &rec,
                        ScatterRecord &srec,
                        curandState *loc) {
  return false;
}
template <MaterialType m>
__host__ bool h_scatter(const SceneObjects &s, const Ray &r,
                        const HitRecord &rec,
                        ScatterRecord &srec) {
  return false;
}
__host__ __device__ bool
scatter_lambertian(const SceneObjects &s, const Ray &r,
                   const HitRecord &rec,
                   ScatterRecord &srec, Vec3 v) {
  Onb uvw;
  uvw.build_from_w(rec.normal);
  auto direction = uvw.local(v);
  srec.pdf_type = COSINE_PDF;
  srec.is_specular = false;
  srec.specular_ray =
      Ray(rec.p, to_unit(direction), r.time());
  srec.pdf_type = COSINE_PDF;
  return true;
}

template <>
__device__ bool
scatter<LAMBERTIAN>(const SceneObjects &s, const Ray &r,
                    const HitRecord &rec,
                    ScatterRecord &srec, curandState *loc) {
  Vec3 rcos = random_cosine_direction(loc);
  srec.attenuation = color_value<TEXTURE>(s, rec, loc);
  return scatter_lambertian(s, r, rec, srec, rcos);
}
template <>
__host__ bool h_scatter<LAMBERTIAN>(const SceneObjects &s,
                                    const Ray &r,
                                    const HitRecord &rec,
                                    ScatterRecord &srec) {
  Vec3 rcos = h_random_cosine_direction();
  srec.attenuation = h_color_value<TEXTURE>(s, rec);
  return scatter_lambertian(s, r, rec, srec, rcos);
}
__host__ __device__ bool
scatter_metal(const SceneObjects &s, const Ray &r,
              const HitRecord &rec, ScatterRecord &srec,
              Vec3 v) {
  int prim_idx;
  float fuzz;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    fuzz = s.g_fuzz_ref_idxs[prim_idx];
  } else {
    prim_idx = rec.primitive_index;
    fuzz = s.fuzz_ref_idxs[prim_idx];
  }

  Vec3 reflected =
      reflect(to_unit(r.direction()), rec.normal);
  srec.specular_ray =
      Ray(rec.p, reflected + fuzz * v, r.time());
  srec.is_specular = true;
  srec.pdf_type = HITTABLE_PDF;
  return true;
}
template <>
__device__ bool
scatter<METAL>(const SceneObjects &s, const Ray &r,
               const HitRecord &rec, ScatterRecord &srec,
               curandState *loc) {
  Vec3 rv = random_in_unit_sphere(loc);
  srec.attenuation = color_value<TEXTURE>(s, rec, loc);
  return scatter_metal(s, r, rec, srec, rv);
}
template <>
__host__ bool h_scatter<METAL>(const SceneObjects &s,
                               const Ray &r,
                               const HitRecord &rec,
                               ScatterRecord &srec) {
  Vec3 rv = h_random_in_unit_sphere();
  srec.attenuation = h_color_value<TEXTURE>(s, rec);
  return scatter_metal(s, r, rec, srec, rv);
}
__host__ __device__ bool
scatter_dielectric(const SceneObjects &s, const Ray &r,
                   const HitRecord &rec,
                   ScatterRecord &srec, float rval) {
  int prim_idx;
  float ref_idx;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    ref_idx = s.g_fuzz_ref_idxs[prim_idx];
  } else {
    prim_idx = rec.primitive_index;
    ref_idx = s.fuzz_ref_idxs[prim_idx];
  }

  srec.attenuation = Vec3(1.0f, 1.0f, 1.0f);
  Vec3 outward_normal;
  Vec3 reflected = reflect(r.direction(), rec.normal);
  //
  float refract_ratio =
      rec.front_face ? (1.0f / ref_idx) : ref_idx;
  Vec3 udir = to_unit(r.direction());
  float cos_theta = dfmin(dot(-udir, rec.normal), 1.0f);
  float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

  bool no_refract = refract_ratio * sin_theta > 1.0f;
  Vec3 dir;
  if (no_refract ||
      fresnelCT(cos_theta, refract_ratio) > rval) {
    // reflect
    dir = reflect(udir, rec.normal);
  } else {
    // refract
    dir = refract(udir, rec.normal, refract_ratio);
  }
  srec.specular_ray = Ray(rec.p, dir, r.time());
  srec.is_specular = true;
  srec.pdf_type = HITTABLE_PDF;
  return true;
}
template <>
__device__ bool
scatter<DIELECTRIC>(const SceneObjects &s, const Ray &r,
                    const HitRecord &rec,
                    ScatterRecord &srec, curandState *loc) {
  float rfloat = curand_uniform(loc);
  return scatter_dielectric(s, r, rec, srec, rfloat);
}
template <>
__host__ bool h_scatter<DIELECTRIC>(const SceneObjects &s,
                                    const Ray &r,
                                    const HitRecord &rec,
                                    ScatterRecord &srec) {
  float rfloat = hrandf();
  return scatter_dielectric(s, r, rec, srec, rfloat);
}
template <>
__device__ bool
scatter<DIFFUSE_LIGHT>(const SceneObjects &s, const Ray &r,
                       const HitRecord &rec,
                       ScatterRecord &srec,
                       curandState *loc) {
  srec.attenuation = color_value<TEXTURE>(s, rec, loc);
  srec.is_specular = false;
  srec.pdf_type = NONE_PDF;
  srec.specular_ray = Ray(rec.p, r.direction(), r.time());
  return false;
}
template <>
__host__ bool
h_scatter<DIFFUSE_LIGHT>(const SceneObjects &s,
                         const Ray &r, const HitRecord &rec,
                         ScatterRecord &srec) {
  srec.attenuation = h_color_value<TEXTURE>(s, rec);
  srec.is_specular = false;
  srec.pdf_type = HITTABLE_PDF;
  srec.specular_ray = Ray(rec.p, r.direction(), r.time());

  return false;
}

template <>
__device__ bool
scatter<ISOTROPIC>(const SceneObjects &s, const Ray &r,
                   const HitRecord &rec,
                   ScatterRecord &srec, curandState *loc) {
  srec.specular_ray =
      Ray(rec.p, random_in_unit_sphere(loc), r.time());
  srec.attenuation = color_value<TEXTURE>(s, rec, loc);
  srec.pdf_type = HITTABLE_PDF;
  srec.is_specular = false;
  return true;
}
template <>
__host__ bool h_scatter<ISOTROPIC>(const SceneObjects &s,
                                   const Ray &r,
                                   const HitRecord &rec,
                                   ScatterRecord &srec) {
  srec.specular_ray =
      Ray(rec.p, h_random_in_unit_sphere(), r.time());
  srec.attenuation = h_color_value<TEXTURE>(s, rec);
  srec.pdf_type = HITTABLE_PDF;
  srec.is_specular = false;
  return true;
}

template <>
__device__ bool
scatter<MATERIAL>(const SceneObjects &s, const Ray &r,
                  const HitRecord &rec, ScatterRecord &srec,
                  curandState *loc) {
  int prim_idx;
  MaterialType mtype;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    mtype = static_cast<MaterialType>(s.g_mtypes[prim_idx]);
  } else {
    prim_idx = rec.primitive_index;
    mtype = static_cast<MaterialType>(s.mtypes[prim_idx]);
  }
  bool res = false;
  if (mtype == LAMBERTIAN) {
    res = scatter<LAMBERTIAN>(s, r, rec, srec, loc);
  } else if (mtype == METAL) {
    res = scatter<METAL>(s, r, rec, srec, loc);
  } else if (mtype == DIELECTRIC) {
    res = scatter<DIELECTRIC>(s, r, rec, srec, loc);
  } else if (mtype == DIFFUSE_LIGHT) {
    res = scatter<DIFFUSE_LIGHT>(s, r, rec, srec, loc);
  } else if (mtype == ISOTROPIC) {
    res = scatter<ISOTROPIC>(s, r, rec, srec, loc);
  }
  // attenuation = Color(1.0f, 1.0f, 0.0f);
  return res;
}

template <>
__host__ bool h_scatter<MATERIAL>(const SceneObjects &s,
                                  const Ray &r,
                                  const HitRecord &rec,
                                  ScatterRecord &srec) {
  int prim_idx;
  MaterialType mtype;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    mtype = static_cast<MaterialType>(s.g_mtypes[prim_idx]);
  } else {
    prim_idx = rec.primitive_index;
    mtype = static_cast<MaterialType>(s.mtypes[prim_idx]);
  }

  bool res = false;
  if (mtype == LAMBERTIAN) {
    res = h_scatter<LAMBERTIAN>(s, r, rec, srec);
  } else if (mtype == METAL) {
    res = h_scatter<METAL>(s, r, rec, srec);
  } else if (mtype == DIELECTRIC) {
    res = h_scatter<DIELECTRIC>(s, r, rec, srec);
  } else if (mtype == DIFFUSE_LIGHT) {
    res = h_scatter<DIFFUSE_LIGHT>(s, r, rec, srec);
  } else if (mtype == ISOTROPIC) {
    res = h_scatter<ISOTROPIC>(s, r, rec, srec);
  }
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
__host__ __device__ float
scattering_pdf(const SceneObjects &s, const Ray &r_in,
               const HitRecord &rec, const Ray &r_out) {
  return 1.0f;
}

template <>
__host__ __device__ float scattering_pdf<LAMBERTIAN>(
    const SceneObjects &s, const Ray &r_in,
    const HitRecord &rec, const Ray &r_out) {
  auto cosine = dot(rec.normal, to_unit(r_out.direction()));
  return cosine < 0 ? 0 : cosine / M_PI;
}
template <>
__host__ __device__ float scattering_pdf<MATERIAL>(
    const SceneObjects &s, const Ray &r_in,
    const HitRecord &rec, const Ray &r_out) {
  int prim_idx;
  MaterialType mtype;
  if (rec.group_scattering) {
    prim_idx = rec.group_index;
    mtype = static_cast<MaterialType>(s.g_mtypes[prim_idx]);
  } else {
    prim_idx = rec.primitive_index;
    mtype = static_cast<MaterialType>(s.mtypes[prim_idx]);
  }

  float res = 1.0f;
  if (mtype == LAMBERTIAN) {
    res = scattering_pdf<LAMBERTIAN>(s, r_in, rec, r_out);
  }
  return res;
}
