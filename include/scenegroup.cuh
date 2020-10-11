#ifndef SCENEGROUP_CUH
#define SCENEGROUP_CUH

#include <mediumc.cuh>
#include <ray.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct SceneGroup {
  // group params
  GroupType gtype;
  int group_size;
  int group_id;
  ScenePrim *prims;
  //
  float density;
  TextureParam tparam;

  __host__ __device__ SceneGroup() {}
  __host__ __device__ SceneGroup(ScenePrim *prm, int gsize,
                                 int gid, GroupType gtp,
                                 float d, TextureParam tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), tparam(tp) {}
  __host__ __device__ inline bool get(int i,
                                      ScenePrim &p) const {
    if (i > 0 && i < group_size) {
      p = prims[i];
      return true;
    }
    return false;
  }
  __host__ __device__ HittableGroup
  to_h_group(Hittable **&hs, Texture *&t) {
    //
    HittableGroup hg;
    switch (gtype) {
    case SOLID: {
      hg = HittableGroup(hs, group_size);
      break;
    }
    case CONSTANT_MEDIUM: {
      HittableGroup hgs(hs, group_size);
      Hittable *h = static_cast<Hittable *>(&hgs);
      ConstantMedium cm(h, density, t);
      Hittable **hss = new Hittable *[1];
      hss[0] = static_cast<Hittable *>(&cm);
      hg = HittableGroup(hss, 1);
      break;
    }
    }
    return hg;
  }
  __host__ __device__ Hittable **
  to_hittable_list(Hittable **&hs) {
    for (int i = 0; i < group_size; i++) {
      ScenePrim pr;
      get(i, pr);
      hs[i] = pr.to_hittable();
    }
    return hs;
  }
  __host__ __device__ Hittable **
  to_hittable_list(unsigned char *&td) {
    Hittable **hs = new Hittable *[group_size];
    for (int i = 0; i < group_size; i++) {
      ScenePrim pr;
      get(i, pr);
      hs[i] = pr.to_hittable(td);
    }
    return hs;
  }
  __device__ Hittable **to_hittable_list(unsigned char *&td,
                                         curandState *loc) {
    Hittable **hs = new Hittable *[group_size];
    for (int i = 0; i < group_size; i++) {
      ScenePrim pr;
      get(i, pr);
      hs[i] = pr.to_hittable(td, loc);
    }
    return hs;
  }
  __device__ Hittable **to_hittable_list(curandState *loc) {
    Hittable **hs = new Hittable *[group_size];
    for (int i = 0; i < group_size; i++) {
      ScenePrim pr;
      get(i, pr);
      hs[i] = pr.to_hittable(loc);
    }
    return hs;
  }
  __host__ __device__ HittableGroup to_hittable() {
    Hittable **hs;
    hs = to_hittable_list(hs);
    Texture *t = tparam.to_texture();
    return to_h_group(hs, t);
  }
  __host__ __device__ HittableGroup
  to_hittable(unsigned char *&td) {
    Hittable **hs = to_hittable_list(td);
    Texture *t = tparam.to_texture(td);
    return to_h_group(hs, t);
  }
  __device__ HittableGroup to_hittable(unsigned char *&td,
                                       curandState *loc) {
    Hittable **hs = to_hittable_list(td, loc);
    Texture *t = tparam.to_texture(td, loc);
    return to_h_group(hs, t);
  }
  __device__ HittableGroup to_hittable(curandState *loc) {
    Hittable **hs = to_hittable_list(loc);
    Texture *t = tparam.to_texture(loc);
    return to_h_group(hs, t);
  }
};

#endif
