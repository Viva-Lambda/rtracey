#pragma once
// scene material
#include <onb.cuh>
#include <ray.cuh>
#include <scenematparam.cuh>
#include <scenetexparam.cuh>
//
#include <record.cuh>
#include <vec3.cuh>
__host__ __device__ float fresnelCT(float costheta,
                                    float ridx) {
  // cook torrence fresnel equation
  float etao = 1 + sqrt(ridx);
  float etau = 1 - sqrt(ridx);
  float eta = etao / etau;
  float g = sqrt(pow(eta, 2) + pow(costheta, 2) - 1);
  float g_c = g - costheta;
  float gplusc = g + costheta;
  float gplus_cc = (gplusc * costheta) - 1;
  float g_cc = (g_c * costheta) + 1;
  float oneplus_gcc = 1 + pow(gplus_cc / g_cc, 2);
  float half_plus_minus = 0.5 * pow(g_c / gplusc, 2);
  return half_plus_minus * oneplus_gcc;
}

__host__ __device__ bool refract(const Vec3 &v,
                                 const Vec3 &n,
                                 float ni_over_nt,
                                 Vec3 &refracted) {
  Vec3 uv = to_unit(v);
  float dt = dot(uv, n);
  float discriminant =
      1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
  if (discriminant > 0) {
    refracted =
        ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
    return true;
  } else {
    return false;
  }
}

__host__ __device__ Vec3 reflect(const Vec3 &v,
                                 const Vec3 &n) {
  return v - 2.0f * dot(v, n) * n;
}

template <class MaT> struct SceneMaterial {
  __device__ static bool
  scatter(MaT m, const Ray &r_in, const HitRecord &rec,
          Vec3 &attenuation, Ray &scattered, float &pdf,
          curandState *local_rand_state) {
    return false;
  }

  __host__ __device__ static float
  scattering_pdf(MaT m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    return 0.0f;
  }
  __host__ __device__ static Color
  emitted(MaT m, float u, float v, const Point3 &p) {
    //
    return Color(0.0f);
  }
};

struct Lambertian {
  TextureParam albedo;
  __host__ __device__ Lambertian(const MaterialParam &mp)
      : albedo(mp.tparam) {}
};
struct Metal {
  TextureParam albedo;
  float fuzz;
  __host__ __device__ Metal(const MaterialParam &mp)
      : albedo(mp.tparam), fuzz(mp.fuzz_ref_idx) {}
};
struct Dielectric {
  float ref_idx;
  __host__ __device__ Dielectric(const MaterialParam &mp)
      : ref_idx(mp.fuzz_ref_idx) {}
};
struct DiffuseLight {
  TextureParam albedo;
  __host__ __device__ DiffuseLight(const MaterialParam &mp)
      : albedo(mp.tparam) {}
};
struct Isotropic {
  TextureParam albedo;
  __host__ __device__ DiffuseLight(const MaterialParam &mp)
      : albedo(mp.tparam) {}
};
