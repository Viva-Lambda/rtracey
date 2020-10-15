#pragma once

#include <ray.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>
// include materials
#include <matparam.cuh>
#include <record.cuh>
#include <scenemat.cuh>

template <> struct SceneMaterial<MaterialParam> {
  __device__ static bool
  scatter(const MaterialParam &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    bool res;
    switch (m.mtype) {
    case LAMBERTIAN: {
      Lambertian mat = m.to_lambert();
      res = SceneMaterial<Lambertian>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
      break;
    }
    case METAL: {
      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
      break;
    }
    case DIELECTRIC: {
      Dielectric mat = m.to_dielectric();
      res = SceneMaterial<Dielectric>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
      break;
    }
    case DIFFUSE_LIGHT: {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
      break;
    }
    case ISOTROPIC: {
      Isotropic mat = m.to_isotropic();
      res = SceneMaterial<Isotropic>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
      break;
    }
    }
    return res;
  }
  __device__ static float
  scattering_pdf(const MaterialParam &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    float res = 0.0f;
    switch (m.mtype) {
    case LAMBERTIAN: {
      Lambertian mat = m.to_lambert();
      res = SceneMaterial<Lambertian>::scattering_pdf(
          mat, r_in, rec, scattered);
      break;
    }
    case METAL: {
      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::scattering_pdf(
          mat, r_in, rec, scattered);
      break;
    }
    case DIELECTRIC: {
      Dielectric mat = m.to_dielectric();
      res = SceneMaterial<Dielectric>::scattering_pdf(
          mat, r_in, rec, scattered);
      break;
    }
    case DIFFUSE_LIGHT: {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::scattering_pdf(
          mat, r_in, rec, scattered);
      break;
    }
    case ISOTROPIC: {
      Isotropic mat = m.to_isotropic();
      res = SceneMaterial<Isotropic>::scattering_pdf(
          mat, r_in, rec, scattered);
      break;
    }
    }
    return res;
  }
  __device__ static Color emitted(const MaterialParam &m,
                                  float u, float v,
                                  const Point3 &p) {
    Color res(0.0f);
    switch (m.mtype) {
    case LAMBERTIAN: {
      Lambertian mat = m.to_lambert();
      res =
          SceneMaterial<Lambertian>::emitted(mat, u, v, p);
      break;
    }
    case METAL: {
      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::emitted(mat, u, v, p);
      break;
    }
    case DIELECTRIC: {
      Dielectric mat = m.to_dielectric();
      res =
          SceneMaterial<Dielectric>::emitted(mat, u, v, p);
      break;
    }
    case DIFFUSE_LIGHT: {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::emitted(mat, u, v,
                                                 p);
      break;
    }
    case ISOTROPIC: {
      Isotropic mat = m.to_isotropic();

      res = SceneMaterial<Isotropic>::emitted(mat, u, v, p);
      break;
    }
    }
    return res;
  }
};
