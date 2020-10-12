#pragma once

#include <scenetexparam.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>
struct MaterialParam {
  TextureParam tparam;
  MaterialType mtype;
  float fuzz_ref_idx;
  __host__ __device__ MaterialParam() {}
  __host__ __device__ MaterialParam(TextureParam tp,
                                    MaterialType mt,
                                    float fri)
      : tparam(tp), mtype(mt), fuzz_ref_idx(fri) {}
};
MaterialParam mkLambertParam(TextureParam t) {
  float f = 0.0f;
  MaterialParam mp(t, LAMBERTIAN, f);
  return mp;
}
MaterialParam mkMetalParam(TextureParam t, float f) {
  MaterialParam mp(t, METAL, f);
  return mp;
}
MaterialParam mkDielectricParam(TextureParam t, float f) {
  MaterialParam mp(t, DIELECTRIC, f);
  return mp;
}
MaterialParam mkDiffuseLightParam(TextureParam t) {
  float f = 0.0f;
  MaterialParam mp(t, DIFFUSE_LIGHT, f);
  return mp;
}
MaterialParam mkIsotropicParam(TextureParam t) {
  float f = 0.0f;
  MaterialParam mp(t, ISOTROPIC, f);
  return mp;
}
