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
  __host__ __device__ MaterialParam()
      : mtype(NONE_MATERIAL), fuzz_ref_idx(0.0f) {}
  __host__ __device__ MaterialParam(const TextureParam &tp,
                                    const MaterialType &mt,
                                    const float fri)
      : tparam(tp), mtype(mt), fuzz_ref_idx(fri) {}
  __host__ __device__ MaterialParam(const MaterialParam &tp)
      : tparam(tp.tparam), mtype(tp.mtype),
        fuzz_ref_idx(tp.fuzz_ref_idx) {}
  __host__ __device__ MaterialParam &
  operator=(const MaterialParam &mp) {
    mtype = mp.mtype;
    tparam = mp.tparam;
    fuzz_ref_idx = mp.fuzz_ref_idx;
    return *this;
  }
};
__host__ __device__ MaterialParam
mkLambertParam(const TextureParam &t) {
  const float f = 0.0f;
  MaterialParam mp(t, LAMBERTIAN, f);
  return mp;
}
__host__ __device__ MaterialParam
mkMetalParam(const TextureParam &t, const float f) {
  MaterialParam mp(t, METAL, f);
  return mp;
}
__host__ __device__ MaterialParam
mkDielectricParam(const TextureParam &t, const float f) {
  MaterialParam mp(t, DIELECTRIC, f);
  return mp;
}
__host__ __device__ MaterialParam
mkDiffuseLightParam(const TextureParam &t) {
  const float f = 0.0f;
  MaterialParam mp(t, DIFFUSE_LIGHT, f);
  return mp;
}
__host__ __device__ MaterialParam
mkIsotropicParam(const TextureParam &t) {
  const float f = 0.0f;
  MaterialParam mp(t, ISOTROPIC, f);
  return mp;
}
