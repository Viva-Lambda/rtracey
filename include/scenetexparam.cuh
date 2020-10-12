#pragma once
#include <scenetype.cuh>
#include <texture.cuh>
#include <vec3.cuh>

struct ImageParam {
  int width, height, bytes_per_pixel, index;
  __host__ __device__ ImageParam() {}
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

  __host__ __device__ TextureParam() {}
  __host__ __device__ TextureParam(TextureType t, Color c,
                                   float s, ImageParam i)
      : ttype(t), cval(c), scale(s), imp(i) {}

  __host__ __device__ SolidColor to_solid() {
    SolidColor sc(cval);
    return sc;
  }
  __host__ __device__ CheckerTexture to_checker() {
    CheckerTexture ct(cval);
    return ct;
  }
  __device__ NoiseTexture to_noise(curandState *loc) {
    NoiseTexture nt(scale, loc);
    return nt;
  }
  __host__ __device__ ImageTexture
  to_image(unsigned char *&td) {
    ImageTexture img(td, imp.width, imp.height,
                     imp.width * imp.bytes_per_pixel,
                     imp.bytes_per_pixel, imp.index);
    return img;
  }
  __host__ __device__ Texture *to_texture() {
    Texture *txt;
    if (ttype == SOLID_COLOR) {
      SolidColor s1 = to_solid();
      txt = static_cast<Texture *>(&s1);
    } else if (ttype == CHECKER) {
      CheckerTexture c1 = to_checker();
      txt = static_cast<Texture *>(&c1);
    }
    return txt;
  }
  __host__ __device__ Texture *
  to_texture(unsigned char *&dt) {
    Texture *txt;
    if (ttype == IMAGE) {
      ImageTexture img = to_image(dt);
      txt = static_cast<Texture *>(&img);
      return txt;
    } else {
      return to_texture();
    }
  }
  __device__ Texture *to_texture(curandState *loc) {
    Texture *txt;
    if (ttype == NOISE) {
      NoiseTexture nt = to_noise(loc);
      txt = static_cast<Texture *>(&nt);
      return txt;
    } else {
      return to_texture();
    }
  }
  __device__ Texture *to_texture(unsigned char *&dt,
                                 curandState *loc) {
    Texture *txt;
    if (ttype == NOISE) {
      NoiseTexture nt = to_noise(loc);
      txt = static_cast<Texture *>(&nt);
      return txt;
    } else {
      return to_texture(dt);
    }
  }
};
TextureParam mkSolidColorParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(SOLID_COLOR, c, s, img);
  return tp;
}
TextureParam mkCheckerTextureParam(Color c) {
  ImageParam img;
  float s = 0.0f;
  TextureParam tp(CHECKER, c, s, img);
  return tp;
}
TextureParam mkNoiseParam(float s) {
  ImageParam img;
  Color c;
  TextureParam tp(NOISE, c, s, img);
  return tp;
}
TextureParam mkImageTextureParam(int w, int h, int bpp,
                                 int idx) {
  ImageParam img(w, h, bpp, idx);
  Color c;
  float s = 0.0f;
  TextureParam tp(IMAGE, c, s, img);
  return tp;
}
