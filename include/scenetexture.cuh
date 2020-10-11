#pragma once
#include <external.hpp>
//
#include <perlin.cuh>
#include <ray.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <TeT> struct SceneTexture {
  __host__ __device__ static Color
  value(TeT txt, float u, float v, const Point3 &p);
};

template <> struct SceneTexture<SolidColor> {
  __host__ __device__ static Color
  value(SolidColor txt, float u, float v, const Point3 &p) {
    //
    return txt.color_value;
  }
};
template <> struct SceneTexture<CheckerTexture> {
  __host__ __device__ static Color value(CheckerTexture txt,
                                         float u, float v,
                                         const Point3 &p) {
    //
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return txt.odd.value(u, v, p);
    } else {
      return txt.even.value(u, v, p);
    }
  }
}
