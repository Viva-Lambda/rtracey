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
    const int mtype = m.mtype;
    if (mtype == LAMBERTIAN) {
      Lambertian mat = m.to_lambert();
      res = SceneMaterial<Lambertian>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
    } else if (mtype == METAL) {
      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
    } else if (mtype == DIELECTRIC) {
      Dielectric mat = m.to_dielectric();
      res = SceneMaterial<Dielectric>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
    } else if (mtype == DIFFUSE_LIGHT) {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
    } else if (mtype == ISOTROPIC) {
      Isotropic mat = m.to_isotropic();
      res = SceneMaterial<Isotropic>::scatter(
          mat, r_in, rec, attenuation, scattered, pdf, loc);
    }
    return res;
  }
  __device__ static float
  scattering_pdf(const MaterialParam &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    float res = 0.0f;
    const int mtype = m.mtype;
    if (mtype == LAMBERTIAN) {
      Lambertian mat = m.to_lambert();
      res = SceneMaterial<Lambertian>::scattering_pdf(
          mat, r_in, rec, scattered);
    } else if (mtype == METAL) {

      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::scattering_pdf(
          mat, r_in, rec, scattered);
    } else if (mtype == DIELECTRIC) {
      Dielectric mat = m.to_dielectric();
      res = SceneMaterial<Dielectric>::scattering_pdf(
          mat, r_in, rec, scattered);
    } else if (mtype == DIFFUSE_LIGHT) {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::scattering_pdf(
          mat, r_in, rec, scattered);
    } else if (mtype == ISOTROPIC) {
      Isotropic mat = m.to_isotropic();
      res = SceneMaterial<Isotropic>::scattering_pdf(
          mat, r_in, rec, scattered);
    }
    return res;
  }
  __device__ static Color emitted(const MaterialParam &m,
                                  float u, float v,
                                  const Point3 &p) {
    Color res(0.0f, 0.0f, 0.0f);
    const int mtype = m.mtype;
    if (mtype == LAMBERTIAN) {
      Lambertian mat = m.to_lambert();
      res =
          SceneMaterial<Lambertian>::emitted(mat, u, v, p);
    } else if (mtype == METAL) {
      Metal mat = m.to_metal();
      res = SceneMaterial<Metal>::emitted(mat, u, v, p);
    } else if (mtype == DIELECTRIC) {
      Dielectric mat = m.to_dielectric();
      res =
          SceneMaterial<Dielectric>::emitted(mat, u, v, p);
    } else if (mtype == DIFFUSE_LIGHT) {
      DiffuseLight mat = m.to_diffuse_light();
      res = SceneMaterial<DiffuseLight>::emitted(mat, u, v,
                                                 p);
    } else if (mtype == ISOTROPIC) {
      Isotropic mat = m.to_isotropic();

      res = SceneMaterial<Isotropic>::emitted(mat, u, v, p);
    }
    return res;
  }
};
