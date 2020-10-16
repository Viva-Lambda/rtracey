#pragma once
#include <scenetexture.cuh>
#include <scenetype.cuh>
#include <texparam.cuh>
#include <texture.cuh>
#include <vec3.cuh>

template <> struct SceneTexture<TextureParam> {
  __device__ static Color value(TextureParam tp, float u,
                                float v, const Point3 &p) {
    Color c(0.0f, 0.0f, 0.0f);
    const int ttype = *tp.ttype;
    if (ttype == NONE_TEXTURE) {
      c = Color(0.0f, 0.0f, 0.0f);
    } else if (ttype == SOLID_COLOR) {
      SolidColor sc = tp.to_solid();
      c = SceneTexture<SolidColor>::value(sc, u, v, p);
    } else if (ttype == CHECKER) {
      CheckerTexture<SolidColor> sc = tp.to_checker();
      c = SceneTexture<CheckerTexture<SolidColor>>::value(
          sc, u, v, p);
    } else if (ttype == NOISE) {
      NoiseTexture sc = tp.to_noise();
      c = SceneTexture<NoiseTexture>::value(sc, u, v, p);
    } else if (ttype == IMAGE) {
      ImageTexture sc = tp.to_image();
      c = SceneTexture<ImageTexture>::value(sc, u, v, p);
    }
    return c;
  }
};
