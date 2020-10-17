#pragma once
#include <external.hpp>
#include <texture.cuh>
#include <utils.cuh>
#include <vec3.cuh>

template <class TeT = Texture> struct SceneTexture {
  __host__ __device__ static Color
  value(TeT txt, float u, float v, const Point3 &p);
};
template <> struct SceneTexture<SolidColor> {
  __host__ __device__ static Color
  value(SolidColor txt, float u, float v, const Point3 &p) {
    return txt.color_value;
  }
};
template <> struct SceneTexture<NoiseTexture> {
  __device__ static Color value(const NoiseTexture &txt,
                                float u, float v,
                                const Point3 &p) {
    float zscale = txt.scale * p.z();
    float turbulance = 10.0f * txt.noise.turb(p);
    Color white(1.0f, 1.0f, 1.0f);
    return white * 0.5f * (1.0f + sin(zscale + turbulance));
  }
};
template <> struct SceneTexture<ImageTexture> {
  __host__ __device__ static Color
  value(const ImageTexture &txt, float u, float v,
        const Point3 &p) {
    if (txt.data == nullptr) {
      return Color(1.0, 0.0, 0.0);
    }
    u = clamp(u, 0.0, 1.0);
    v = 1.0 - clamp(v, 0.0, 1.0); // flip v to im coords
    int w = txt.width;
    int h = txt.height;
    int xi = (int)(u * w);
    int yj = (int)(v * h);
    xi = xi >= w ? w - 1 : xi;
    yj = yj >= h ? h - 1 : yj;

    //
    int bpp = txt.bytes_per_pixel;
    int bytes_per_line = bpp * w;
    int idx = txt.index;
    int pixel = yj * bytes_per_line + xi * bpp + idx;
    Color c(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < bpp; i++) {
      float pixel_val =
          static_cast<float>(txt.data[pixel + i]);
      c[i] = pixel_val / 255;
    }
    return c;
  }
};
template <>
struct SceneTexture<CheckerTexture<SolidColor>> {
  __host__ __device__ static Color
  value(CheckerTexture<SolidColor> txt, float u, float v,
        const Point3 &p) {
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return SceneTexture<SolidColor>::value(txt.odd, u, v,
                                             p);
    } else {
      return SceneTexture<SolidColor>::value(txt.even, u, v,
                                             p);
    }
  }
};
template <>
struct SceneTexture<CheckerTexture<NoiseTexture>> {
  __device__ static Color
  value(CheckerTexture<NoiseTexture> txt, float u, float v,
        const Point3 &p) {
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return SceneTexture<NoiseTexture>::value(txt.odd, u,
                                               v, p);
    } else {
      return SceneTexture<NoiseTexture>::value(txt.even, u,
                                               v, p);
    }
  }
};
template <>
struct SceneTexture<CheckerTexture<ImageTexture>> {
  __host__ __device__ static Color
  value(CheckerTexture<ImageTexture> txt, float u, float v,
        const Point3 &p) {
    float sines = sin(10 * p.x()) * sin(10.0f * p.y()) *
                  sin(10.0f * p.z());
    if (sines < 0) {
      return SceneTexture<ImageTexture>::value(txt.odd, u,
                                               v, p);
    } else {
      return SceneTexture<ImageTexture>::value(txt.even, u,
                                               v, p);
    }
  }
};
