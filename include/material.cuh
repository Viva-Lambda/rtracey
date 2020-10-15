#pragma once
#include <texparam.cuh>
struct Material {};

struct Lambertian : Material {
  TextureParam albedo;
  __host__ __device__ Lambertian(const TextureParam &mp)
      : albedo(mp) {}
};
struct Metal : Material {
  TextureParam albedo;
  float fuzz;
  __host__ __device__ Metal(const TextureParam &mp, float f)
      : albedo(mp), fuzz(f) {}
};
struct Dielectric : Material {
  float ref_idx;
  __host__ __device__ Dielectric(float f) : ref_idx(f) {}
};
struct DiffuseLight : Material {
  TextureParam albedo;
  __host__ __device__ DiffuseLight(const TextureParam &mp)
      : albedo(mp) {}
};
struct Isotropic : Material {
  TextureParam albedo;
  __host__ __device__ Isotropic(const TextureParam &mp)
      : albedo(mp) {}
};
