#pragma once
// scene material
#include <material.cuh>
#include <onb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <scenetexparam.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <class MaT> struct SceneMaterial {
  __device__ static bool
  scatter(const MaT &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf,
          curandState *local_rand_state) {
    return false;
  }
  __device__ static float
  scattering_pdf(const MaT &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
  __device__ static Color
  emitted(const MaT &m, float u, float v, const Point3 &p) {
    return Color(0.0f, 0.0f, 0.0f);
  }
};
template <> struct SceneMaterial<Lambertian> {
  __device__ static bool
  scatter(const Lambertian &lamb, const Ray &r_in,
          const HitRecord &rec, Color &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    Onb uvw;
    uvw.build_from_w(rec.normal);
    auto direction =
        uvw.local(random_cosine_direction(loc));
    scattered = Ray(rec.p, to_unit(direction), r_in.time());
    attenuation = SceneTexture<TextureParam>::value(
        lamb.albedo, rec.u, rec.v, rec.p);
    pdf = dot(uvw.w(), scattered.direction()) / M_PI;
    return true;
  }
  __device__ static float
  scattering_pdf(const Lambertian &lamb, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    auto cosine =
        dot(rec.normal, to_unit(scattered.direction()));
    return cosine < 0 ? 0 : cosine / M_PI;
  }
  __device__ static Color emitted(const Lambertian &m,
                                  float u, float v,
                                  const Point3 &p) {
    return Color(0.0f, 0.0f, 0.0f);
  }
};
template <> struct SceneMaterial<Metal> {
  __device__ static bool
  scatter(const Metal &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    Vec3 reflected =
        reflect(to_unit(r_in.direction()), rec.normal);
    scattered =
        Ray(rec.p,
            reflected + m.fuzz * random_in_unit_sphere(loc),
            r_in.time());
    attenuation = SceneTexture<TextureParam>::value(
        m.albedo, rec.u, rec.v, rec.p);
    pdf = 1.0f;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }
  __device__ static float
  scattering_pdf(const Metal &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
  __device__ static Color emitted(const Metal &m, float u,
                                  float v,
                                  const Point3 &p) {
    return Color(0.0f, 0.0f, 0.0f);
  }
};
template <> struct SceneMaterial<Dielectric> {
  __device__ static bool
  scatter(const Dielectric &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    pdf = 1.0f;
    Vec3 outward_normal;
    Vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
      outward_normal = -rec.normal;
      ni_over_nt = m.ref_idx;
      cosine = dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
      cosine = sqrt(1.0f -
                    ni_over_nt * ni_over_nt *
                        (1 - cosine * cosine));
    } else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / m.ref_idx;
      cosine = -dot(r_in.direction(), rec.normal) /
               r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal,
                ni_over_nt, refracted))
      reflect_prob = fresnelCT(cosine, m.ref_idx);
    else
      reflect_prob = 1.0f;
    if (curand_uniform(loc) < reflect_prob)
      scattered = Ray(rec.p, reflected, r_in.time());
    else
      scattered = Ray(rec.p, refracted, r_in.time());
    return true;
  }
  __device__ static float
  scattering_pdf(const Dielectric &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
  __device__ static Color emitted(const Dielectric &m,
                                  float u, float v,
                                  const Point3 &p) {
    return Color(0.0f, 0.0f, 0.0f);
  }
};
template <> struct SceneMaterial<DiffuseLight> {
  __device__ static bool
  scatter(const DiffuseLight &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    pdf = 1.0f;
    return false;
  }
  __device__ static Color emitted(const DiffuseLight &m,
                                  float u, float v,
                                  const Point3 &p) {
    return SceneTexture<TextureParam>::value(m.albedo, u, v,
                                             p);
  }
  __device__ static float
  scattering_pdf(const DiffuseLight &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
};
template <> struct SceneMaterial<Isotropic> {
  __device__ static bool
  scatter(const Isotropic &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    scattered =
        Ray(rec.p, random_in_unit_sphere(loc), r_in.time());
    attenuation = SceneTexture<TextureParam>::value(
        m.albedo, rec.u, rec.v, rec.p);
    pdf = 1.0f;
    return true;
  }
  __device__ static float
  scattering_pdf(const Isotropic &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
  __device__ static Color emitted(const Isotropic &m,
                                  float u, float v,
                                  const Point3 &p) {
    return Color(0.0f, 0.0f, 0.0f);
  }
};
