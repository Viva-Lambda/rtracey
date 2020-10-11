#pragma once
#include <aarect.cuh>
#include <hittable.cuh>
#include <material.cuh>
#include <ray.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <sphere.cuh>
#include <texture.cuh>
#include <triangle.cuh>
#include <vec3.cuh>

struct ScenePrim {
  // texture params
  TextureType ttype;
  Color cval;
  float scale;

  // image params
  int width;
  int height;
  int bytes_per_pixel;
  int image_index;

  // material params
  MaterialType mtype;
  float fuzz_ref_idx;

  // hittable params
  HittableType htype;
  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float radius;
  float n1x, n1y, n1z;

  // group params
  int group_id;
  int group_index;

  __host__ __device__ ScenePrim() {}
  __host__ __device__ ScenePrim(MaterialParam mt,
                                HittableParam ht,
                                int gindex, int gid)
      : ttype(mt.tparam.ttype), cval(mt.tparam.cval),
        scale(mt.tparam.scale), width(mt.tparam.imp.width),
        height(mt.tparam.imp.height),
        bytes_per_pixel(mt.tparam.imp.bytes_per_pixel),
        image_index(mt.tparam.imp.index), mtype(mt.mtype),
        fuzz_ref_idx(mt.fuzz_ref_idx), p1x(ht.p1x),
        p1y(ht.p1y), p1z(ht.p1z), p2x(ht.p2x), p2y(ht.p2y),
        p2z(ht.p2z), n1x(ht.n1x), n1y(ht.n1y), n1z(ht.n1z),
        radius(ht.radius), htype(ht.htype),
        group_index(gindex), group_id(gid) {}

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
    ImageTexture img(td, width, height,
                     width * bytes_per_pixel,
                     bytes_per_pixel, image_index);
    return img;
  }
  __host__ __device__ Lambertian to_lambert(Texture *&t) {
    Lambertian lamb(t);
    return lamb;
  }
  __host__ __device__ Lambertian to_lambert(Color c) {
    Lambertian lamb(c);
    return lamb;
  }
  __host__ __device__ Metal to_metal(Color c) {
    Metal mm(c, fuzz_ref_idx);
    return mm;
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
  to_diffuse_light(Color c) {
    DiffuseLight dl(c);
    return dl;
  }
  __host__ __device__ DiffuseLight
  to_diffuse_light(Texture *&t) {
    DiffuseLight dl(t);
    return dl;
  }
  __host__ __device__ Isotropic to_isotropic(Color c) {
    Isotropic iso(c);
    return iso;
  }
  __host__ __device__ Isotropic to_isotropic(Texture *&t) {
    Isotropic iso(t);
    return iso;
  }
  __host__ __device__ Sphere to_sphere(Material *&mt) {
    Point3 cent(p1x, p1y, p1z);
    Sphere sp(cent, radius, mt);
    return sp;
  }
  __host__ __device__ MovingSphere
  to_moving_sphere(Material *&mt) {
    Point3 cent1(p1x, p1y, p1z);
    Point3 cent2(p2x, p2y, p2z);
    MovingSphere sp(cent1, cent2, n1x, n1y, radius, mt);
    return sp;
  }
  __host__ __device__ Triangle to_triangle(Material *&mt) {
    Point3 p1(p1x, p1y, p1z);
    Point3 p2(p2x, p2y, p2z);
    Point3 p3(n1x, n1y, n1z);
    Triangle tri(p1, p2, p3, mt);
    return tri;
  }
  __host__ __device__ void rect_val(float &a0, float &a1,
                                    float &b0, float &b1,
                                    float &k) {
    a0 = p1x;
    a1 = p1y;
    b0 = p2x;
    b1 = p2y;
    k = radius;
  }
  __host__ __device__ XYRect to_xyrect(Material *&mt) {
    //
    float x0, x1, y0, y1, k;
    rect_val(x0, x1, y0, y1, k);
    XYRect xyr(x0, x1, y0, y1, k, mt);
    return xyr;
  }
  __host__ __device__ XZRect to_xzrect(Material *&mt) {
    //
    float x0, x1, z0, z1, k;
    rect_val(x0, x1, z0, z1, k);
    XZRect xzr(x0, x1, z0, z1, k, mt);
    return xzr;
  }
  __host__ __device__ YZRect to_yzrect(Material *&mt) {
    //
    float y0, y1, z0, z1, k;
    rect_val(y0, y1, z0, z1, k);
    YZRect yzr(y0, y1, z0, z1, k, mt);
    return yzr;
  }
  __host__ __device__ Hittable *to_hittable(Texture *&tx,
                                            Material *&mt) {
    Hittable *ht;
    if (ttype == SOLID_COLOR) {
      SolidColor s1 = to_solid();
      tx = static_cast<Texture *>(&s1);
    } else if (ttype == CHECKER) {
      CheckerTexture c1 = to_checker();
      tx = static_cast<Texture *>(&c1);
    }
    switch (mtype) {
    case LAMBERTIAN: {
      Lambertian lamb = to_lambert(tx);
      mt = static_cast<Material *>(&lamb);
      break;
    }
    case METAL: {
      Metal met = to_metal(tx);
      mt = static_cast<Material *>(&met);
      break;
    }
    case DIELECTRIC: {
      Dielectric diel = to_dielectric();
      mt = static_cast<Material *>(&diel);
      break;
    }
    case DIFFUSE_LIGHT: {
      DiffuseLight dl = to_diffuse_light(tx);
      mt = static_cast<Material *>(&dl);
      break;
    }
    case ISOTROPIC: {
      Isotropic isot = to_isotropic(tx);
      mt = static_cast<Material *>(&isot);
      break;
    }
    }
    switch (htype) {
    case TRIANGLE: {
      Triangle tri = to_triangle(mt);
      ht = static_cast<Hittable *>(&tri);
      break;
    }
    case SPHERE: {
      Sphere sp = to_sphere(mt);
      ht = static_cast<Hittable *>(&sp);
      break;
    }
    case MOVING_SPHERE: {
      MovingSphere sp = to_moving_sphere(mt);
      ht = static_cast<Hittable *>(&sp);
      break;
    }
    case XY_RECT: {
      XYRect xyr = to_xyrect(mt);
      ht = static_cast<Hittable *>(&xyr);
      break;
    }
    case XZ_RECT: {
      XZRect xzr = to_xzrect(mt);
      ht = static_cast<Hittable *>(&xzr);
      break;
    }
    case YZ_RECT: {
      YZRect yzr = to_yzrect(mt);
      ht = static_cast<Hittable *>(&yzr);
      break;
    }
    }
    return ht;
  }
  __host__ __device__ Hittable *
  to_hittable(unsigned char *&dt, Texture *&tx,
              Material *&mt) {
    if (ttype == IMAGE) {
      ImageTexture img = to_image(dt);
      tx = static_cast<Texture *>(&img);
    }
    return to_hittable(tx, mt);
  }
  __device__ Hittable *to_hittable(Texture *&tx,
                                   Material *&mt,
                                   curandState *loc) {
    if (ttype == NOISE) {
      NoiseTexture nt = to_noise(loc);
      tx = static_cast<Texture *>(&nt);
    }
    return to_hittable(tx, mt);
  }
  __device__ Hittable *to_hittable(unsigned char *&dt,
                                   Texture *&tx,
                                   Material *&mt,
                                   curandState *loc) {
    if (ttype == NOISE) {
      NoiseTexture nt = to_noise(loc);
      tx = static_cast<Texture *>(&nt);
    }
    return to_hittable(dt, tx, mt);
  }
};
