#pragma once

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
  __host__ __device__ HittableGroup to_hittable() {
    Hittable **hs = new Hittable *[group_size];
    for (int i = 0; i < group_size; i++) {
      ScenePrim pr;
      get(i, pr);
      Texture *txt;
      Material *mt;
      hs[i] = pr.to_hittable(txt, mt);
    }
    switch (gtype) {
    case SOLID: {
      HittableGroup hg(hs, group_size);
      break;
    }
    case CONSTANT_MEDIUM: {
      HittableGroup hg(hs, group_size);
      break;
    }
    }
  }
};
