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

struct MaterialParam {
  TextureParam tparam;
  MaterialType mtype;
  float fuzz_ref_idx;
  __host__ __device__ MaterialParam(TextureParam tp,
                                    MaterialType mt,
                                    float fri)
      : tparam(tp), mtype(mt), fuzz_ref_idx(fri) {}
  __host__ __device__ Lambertian to_lambert(Texture *t) {
    Lambertian lamb(t);
    return lamb;
  }
  __host__ __device__ Metal to_metal(Texture *&t) {
    Metal mm(t, fuzz_ref_idx);
    return mm;
  }
  __host__ __device__ Dielectric to_dielectric() {
    Dielectric dd(fuzz_ref_idx);
    return dd;
  }
  __host__ __device__ DiffuseLight
  to_diffuse_light(Texture *&t) {
    DiffuseLight dl(t);
    return dl;
  }
  __host__ __device__ Isotropic to_isotropic(Texture *&t) {
    Isotropic iso(t);
    return iso;
  }
  __host__ __device__ Material *to_material() {
    Texture *t = tparam.to_texture();
    return to_material(t);
  }
  __host__ __device__ Material *to_material(Texture *&txt) {
    Material *mt;
    switch (mtype) {
    case LAMBERTIAN: {
      Lambertian lamb = to_lambert(txt);
      mt = static_cast<Material *>(&lamb);
      break;
    }
    case METAL: {
      Metal met = to_metal(txt);
      mt = static_cast<Material *>(&met);
      break;
    }
    case DIELECTRIC: {
      Dielectric diel = to_dielectric();
      mt = static_cast<Material *>(&diel);
      break;
    }
    case DIFFUSE_LIGHT: {
      DiffuseLight dl = to_diffuse_light(txt);
      mt = static_cast<Material *>(&dl);
      break;
    }
    case ISOTROPIC: {
      Isotropic isot = to_isotropic(txt);
      mt = static_cast<Material *>(&isot);
      break;
    }
    }
    return mt;
  }
  __host__ __device__ Material *
  to_material(unsigned char *&td) {
    Texture *t = tparam.to_texture(td);
    return to_material(t);
  }
  __device__ Material *to_material(curandState *td) {
    Texture *t = tparam.to_texture(td);
    return to_material(t);
  }
  __device__ Material *to_material(unsigned char *&td,
                                   curandState *loc) {
    Texture *t = tparam.to_texture(td, loc);
    return to_material(t);
  }
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

struct HittableParam {
  HittableType htype;

  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float radius;
  float n1x, n1y, n1z;
  __host__ __device__ HittableParam() {}
  __host__ __device__ HittableParam(HittableType ht,
                                    float _p1x, float _p1y,
                                    float _p1z, float _p2x,
                                    float _p2y, float _p2z,
                                    float _n1x, float _n1y,
                                    float _n1z, float r)
      : htype(ht), p1x(_p1x), p1y(_p1y), p1z(_p1z),
        p2x(_p2x), p2y(_p2y), p2z(_p2z), n1x(_n1x),
        n1y(_n1y), n1z(_n1z), radius(r) {}
};

HittableParam mkRectHittable(float a0, float a1, float b0,
                             float b1, Vec3 anormal,
                             float k) {
  HittableParam param;
  param.p1x = a0;
  param.p1y = a1;
  param.p2x = b0;
  param.p2y = b1;
  param.radius = k;
  if (anormal.z() == 1) {
    param.htype = XY_RECT;
  } else if (anormal.y() == 1) {
    param.htype = XZ_RECT;
  } else if (anormal.x() == 1) {
    param.htype = YZ_RECT;
  }
  return param;
}
HittableParam mkYZRectHittable(float a0, float a1, float b0,
                               float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(1, 0, 0), k);
}
HittableParam mkXZRectHittable(float a0, float a1, float b0,
                               float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(0, 1, 0), k);
}
HittableParam mkXYRectHittable(float a0, float a1, float b0,
                               float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(0, 0, 1), k);
}

HittableParam mkSphereHittable(Point3 cent, float rad) {
  HittableParam param;
  param.htype = SPHERE;
  param.p1x = cent.x();
  param.p1y = cent.y();
  param.p1z = cent.z();
  param.radius = rad;
  return param;
}
HittableParam mkMovingSphereHittable(Point3 cent1,
                                     Point3 cent2,
                                     float rad, float t0,
                                     float t1) {
  HittableParam param;
  param.htype = MOVING_SPHERE;
  param.p1x = cent1.x();
  param.p1y = cent1.y();
  param.p1z = cent1.z();
  param.p2x = cent2.x();
  param.p2y = cent2.y();
  param.p2z = cent2.z();
  param.radius = rad;
  param.n1x = t0;
  param.n1y = t1;
  return param;
}
