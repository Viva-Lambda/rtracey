#pragma once
#include <scenetype.cuh>
#include <texparam.cuh>
#include <vec3.cuh>
// include materials
#include <material.cuh>

struct MaterialParam {
  TextureParam tparam;
  MaterialType mtype;
  float fuzz_ref_idx;
  __host__ __device__ MaterialParam() {}
  __host__ __device__ MaterialParam(TextureParam tp,
                                    MaterialType mt,
                                    float fri)
      : tparam(tp), mtype(mt), fuzz_ref_idx(fri) {}
  __host__ __device__ Lambertian to_lambert() const {
    Lambertian lc(tparam);
    return lc;
  }
  __host__ __device__ Metal to_metal() const {
    Metal met(tparam, fuzz_ref_idx);
    return met;
  }
  __host__ __device__ Dielectric to_dielectric() const {
    Dielectric diel(fuzz_ref_idx);
    return diel;
  }
  __host__ __device__ DiffuseLight
  to_diffuse_light() const {
    DiffuseLight df(tparam);
    return df;
  }
  __host__ __device__ Isotropic to_isotropic() const {
    Isotropic isot(tparam);
    return isot;
  }
};
__host__ __device__ MaterialParam mkLambertParam(const TextureParam &t) {
  float f = 0.0f;
  MaterialParam mp(t, LAMBERTIAN, f);
  return mp;
}
__host__ __device__ MaterialParam mkMetalParam(const TextureParam &t, float f) {
  MaterialParam mp(t, METAL, f);
  return mp;
}
__host__ __device__ MaterialParam mkDielectricParam(const TextureParam &t,
                                float f) {
  MaterialParam mp(t, DIELECTRIC, f);
  return mp;
}
__host__ __device__ MaterialParam mkDiffuseLightParam(const TextureParam &t) {
  float f = 0.0f;
  MaterialParam mp(t, DIFFUSE_LIGHT, f);
  return mp;
}
__host__ __device__ MaterialParam mkIsotropicParam(const TextureParam &t) {
  float f = 0.0f;
  MaterialParam mp(t, ISOTROPIC, f);
  return mp;
}
