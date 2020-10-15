#pragma once
#include <external.hpp>
#include <perlin.cuh>
#include <ray.cuh>
#include <utils.cuh>
#include <vec3.cuh>

struct Texture {};
struct SolidColor : Texture {
  Color color_value;
  __host__ __device__ SolidColor(Color c)
      : color_value(c) {}

  __host__ __device__ SolidColor(float red, float green,
                                 float blue)
      : color_value(red, green, blue) {}
};
template <class TeX = Texture> struct CheckerTexture {
  TeX odd;
  TeX even;
  __host__ __device__ CheckerTexture(const TeX &c1,
                                     const TeX &c2)
      : odd(c1), even(c2) {}
};
template <> struct CheckerTexture<SolidColor> {
  SolidColor odd;
  SolidColor even;
  __host__ __device__ CheckerTexture(const SolidColor &c1,
                                     const SolidColor &c2)
      : odd(c1), even(c2) {}
  __host__ __device__ CheckerTexture(Color c1, Color c2)
      : odd(SolidColor(c1)), even(SolidColor(c2)) {}
  __host__ __device__ CheckerTexture(Color c1)
      : odd(SolidColor(c1)), even(SolidColor(1.0f - c1)) {}
  __host__ __device__ CheckerTexture(const SolidColor &c1)
      : odd(c1), even(SolidColor(1.0f - c1.color_value)) {}
};
struct NoiseTexture : Texture {
  float scale;
  Perlin noise;

  __host__ __device__ NoiseTexture() {}
  __device__ NoiseTexture(float s, curandState *loc)
      : scale(s), noise(Perlin(loc)) {}
};
struct ImageTexture : Texture {
  unsigned char *data;
  int width, height;
  int bytes_per_line; // == bytes_per_pixel * width;
  int bytes_per_pixel;
  int index;
  __host__ __device__ ImageTexture()
      : data(nullptr), width(0), height(0),
        bytes_per_line(0), bytes_per_pixel(0), index(0) {}
  __host__ __device__ ImageTexture(int w, int h, int bpl,
                                   int bpp, int ind)
      : data(nullptr), width(w), height(h),
        bytes_per_line(bpl), bytes_per_pixel(bpp),
        index(ind) {}
  __host__ __device__ ImageTexture(int w, int h, int bpp,
                                   int ind)
      : data(nullptr), width(w), height(h),
        bytes_per_line(w * bpp), bytes_per_pixel(bpp),
        index(ind) {}
  __host__ __device__ void set_data(unsigned char *&td) {
    data = td;
  }
};
template <> struct CheckerTexture<ImageTexture> {
  ImageTexture odd;
  ImageTexture even;
  __host__ __device__ CheckerTexture(const ImageTexture &c1,
                                     const ImageTexture &c2)
      : odd(c1), even(c2) {}
};
