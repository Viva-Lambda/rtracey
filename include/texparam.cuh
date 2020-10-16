#pragma once
#include <texture.cuh>
#include <vec3.cuh>

struct ImageParam {
  const int *width, *height, *bytes_per_pixel, *index;
  __host__ __device__ ImageParam()
      : width(nullptr), height(nullptr),
        bytes_per_pixel(nullptr), index(nullptr) {}
  __host__ __device__ ImageParam(const int *w, const int *h,
                                 const int *bpp,
                                 const int *i)
      : width(w), height(h), bytes_per_pixel(bpp),
        index(i) {}
};
struct TextureParam {
  const int *ttype;
  const float *tp1x, *tp1y, *tp1z;
  const float *scale;
  const int *width, *height, *bytes_per_pixel, *index;
  unsigned char *tdata;
  curandState *loc;

  __host__ __device__ TextureParam()
      : tdata(nullptr), scale(nullptr), tp1x(nullptr),
        tp1y(nullptr), tp1z(nullptr), loc(nullptr),
        ttype(nullptr), width(nullptr), height(nullptr),
        bytes_per_pixel(nullptr), index(nullptr) {}

  __host__ __device__
  TextureParam(const int *texture_type, const float *_tp1x,
               const float *_tp1y, const float *_tp1z,
               const float *s, const int *w, const int *h,
               const int *bpp, const int *i)
      : tdata(nullptr), loc(nullptr), ttype(texture_type),
        tp1x(_tp1x), tp1y(_tp1y), tp1z(_tp1z), width(w),
        height(h), bytes_per_pixel(bpp), index(i),
        scale(s) {}
  __host__ __device__ TextureParam(const int *texture_type,
                                   const float *_tp1x,
                                   const float *_tp1y,
                                   const float *_tp1z,
                                   const float *s,
                                   const ImageParam &imp)
      : tdata(nullptr), loc(nullptr), ttype(texture_type),
        tp1x(_tp1x), tp1y(_tp1y), tp1z(_tp1z),
        width(imp.width), height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel),
        index(imp.index), scale(s) {}

  __host__ __device__ TextureParam(unsigned char *td,
                                   const int *texture_type,
                                   const float *_tp1x,
                                   const float *_tp1y,
                                   const float *_tp1z,
                                   const float *s,
                                   const ImageParam &imp)
      : tdata(td), loc(nullptr), ttype(texture_type),
        tp1x(_tp1x), tp1y(_tp1y), tp1z(_tp1z),
        width(imp.width), height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel),
        index(imp.index), scale(s) {}

  __device__
  TextureParam(unsigned char *td, curandState *lc,
               const int *texture_type, const float *_tp1x,
               const float *_tp1y, const float *_tp1z,
               const float *s, const ImageParam &imp)
      : tdata(td), loc(lc), ttype(texture_type),
        tp1x(_tp1x), tp1y(_tp1y), tp1z(_tp1z),
        width(imp.width), height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel),
        index(imp.index), scale(s) {}

  __device__
  TextureParam(curandState *lc, const int *texture_type,
               const float *_tp1x, const float *_tp1y,
               const float *_tp1z, const float *s,
               const ImageParam &imp)
      : tdata(nullptr), loc(lc), ttype(texture_type),
        tp1x(_tp1x), tp1y(_tp1y), tp1z(_tp1z),
        width(imp.width), height(imp.height),
        bytes_per_pixel(imp.bytes_per_pixel),
        index(imp.index), scale(s) {}

  __host__ __device__ TextureParam(const TextureParam &tp)
      : ttype(tp.ttype), tp1x(tp.tp1x), tp1y(tp.tp1y),
        tp1z(tp.tp1z), scale(tp.scale), width(tp.width),
        height(tp.height),
        bytes_per_pixel(tp.bytes_per_pixel),
        index(tp.index), tdata(tp.tdata), loc(tp.loc) {}

  __host__ __device__ SolidColor to_solid() {
    SolidColor sc(tp1x, tp1y, tp1z);
    return sc;
  }
  __host__ __device__ CheckerTexture<SolidColor>
  to_checker() {
    CheckerTexture<SolidColor> ct(
        Color((float *)tp1x, (float *)tp1y, (float *)tp1z));
    return ct;
  }
  __device__ NoiseTexture to_noise() {
    NoiseTexture nt(scale, loc);
    return nt;
  }
  __host__ __device__ ImageTexture to_image() {
    ImageTexture img(width, height, bytes_per_pixel, index,
                     tdata);
    return img;
  }
};
__host__ __device__ TextureParam
mkSolidColorParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(&SOLID_COLOR, c.e1, c.e2, c.e3, &s, img);
  return tp;
}
__host__ __device__ TextureParam
mkCheckerTextureParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(&CHECKER, c.e1, c.e2, c.e3, &s, img);
  return tp;
}
__host__ __device__ TextureParam mkNoiseParam(float s) {
  ImageParam img;
  Color c;
  TextureParam tp(&NOISE, c.e1, c.e2, c.e3, &s, img);
  return tp;
}
__host__ __device__ TextureParam
mkImageTextureParam(int w, int h, int bpp, int idx) {
  ImageParam img(&w, &h, &bpp, &idx);
  Color c;
  float s = 0.0f;
  TextureParam tp(&IMAGE, c.e1, c.e2, c.e3, &s, img);
  return tp;
}
