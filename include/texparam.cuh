#pragma once
#include <texture.cuh>
#include <vec3.cuh>

struct ImageParam {
  int width, height, bytes_per_pixel, index;
  __host__ __device__ ImageParam()
      : width(0), height(0), bytes_per_pixel(0), index(0) {}
  __host__ __device__ ImageParam(int w, int h, int bpp,
                                 int i)
      : width(w), height(h), bytes_per_pixel(bpp),
        index(i) {}
};
struct TextureParam {
  TextureType ttype;
  Color cval;
  float scale;
  ImageParam imp;
  unsigned char *tdata;
  curandState *loc;

  __host__ __device__ TextureParam()
      : tdata(nullptr), scale(0.0f), cval(Color(0.0f)),
        loc(nullptr), ttype(NONE_TEXTURE) {}
  __host__ __device__ TextureParam(TextureType t, Color c,
                                   float s, ImageParam i)
      : ttype(t), cval(c), scale(s), imp(i), tdata(nullptr),
        loc(nullptr) {}
  __host__ __device__ TextureParam(unsigned char *td,
                                   TextureType t, Color c,
                                   float s, ImageParam i)
      : tdata(td), ttype(t), cval(c), scale(s), imp(i),
        loc(nullptr) {}
  __device__ TextureParam(unsigned char *td,
                          curandState *lc, TextureType t,
                          Color c, float s, ImageParam i)
      : tdata(td), ttype(t), cval(c), scale(s), imp(i),
        loc(lc) {}
  __device__ TextureParam(curandState *lc, TextureType t,
                          Color c, float s, ImageParam i)
      : tdata(nullptr), ttype(t), cval(c), scale(s), imp(i),
        loc(lc) {}
  __device__ void set_curand_loc(curandState *lc) {
    loc = lc;
  }

  __host__ __device__ SolidColor to_solid() {
    SolidColor sc(cval);
    return sc;
  }
  __host__ __device__ CheckerTexture<SolidColor>
  to_checker() {
    CheckerTexture<SolidColor> ct(cval);
    return ct;
  }
  __device__ NoiseTexture to_noise() {
    NoiseTexture nt(scale, loc);
    return nt;
  }
  __host__ __device__ ImageTexture to_image() {
    ImageTexture img(imp.width, imp.height,
                     imp.width * imp.bytes_per_pixel,
                     imp.bytes_per_pixel, imp.index);
    img.set_data(tdata);
    return img;
  }
};
__host__ __device__ TextureParam
mkSolidColorParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(SOLID_COLOR, c, s, img);
  return tp;
}
__host__ __device__ TextureParam
mkCheckerTextureParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(CHECKER, c, s, img);
  return tp;
}
__host__ __device__ TextureParam mkNoiseParam(float s) {
  ImageParam img;
  Color c;
  TextureParam tp(NOISE, c, s, img);
  return tp;
}
__host__ __device__ TextureParam
mkImageTextureParam(int w, int h, int bpp, int idx) {
  ImageParam img(w, h, bpp, idx);
  Color c;
  float s = 0.0f;
  TextureParam tp(IMAGE, c, s, img);
  return tp;
}
