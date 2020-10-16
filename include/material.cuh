#pragma once
#include <texparam.cuh>
struct Material {};

struct Lambertian : Material {
  const TextureParam albedo;
  __host__ __device__ Lambertian(const TextureParam &mp)
      : albedo(mp) {}
};
struct Metal : Material {
  const TextureParam albedo;
  const float *fuzz;
  __host__ __device__ Metal(const TextureParam &mp,
                            const float *f)
      : albedo(mp), fuzz(f) {}
};
struct Dielectric : Material {
  const float *ref_idx;
  __host__ __device__ Dielectric(const float *f)
      : ref_idx(f) {}
};
struct DiffuseLight : Material {
  const TextureParam albedo;
  __host__ __device__ DiffuseLight(const TextureParam &mp)
      : albedo(mp) {}
};
struct Isotropic : Material {
  const TextureParam albedo;
  __host__ __device__ Isotropic(const TextureParam &mp)
      : albedo(mp) {}
};
