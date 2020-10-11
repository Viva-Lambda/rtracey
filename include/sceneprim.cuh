#ifndef SCENEPRIM_CUH
#define SCENEPRIM_CUH

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

  __host__ __device__ ImageParam to_img_param() {
    ImageParam imp(width, height, bytes_per_pixel,
                   image_index);
    return imp;
  }
  __host__ __device__ TextureParam to_texture_param() {
    ImageParam imp = to_img_param();
    TextureParam tp(ttype, cval, scale, imp);
    return tp;
  }
  __host__ __device__ MaterialParam to_material_param() {
    TextureParam tp = to_texture_param();
    MaterialParam mp(tp, mtype, fuzz_ref_idx);
    return mp;
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
    float x0, x1, y0, y1, k;
    rect_val(x0, x1, y0, y1, k);
    XYRect xyr(x0, x1, y0, y1, k, mt);
    return xyr;
  }
  __host__ __device__ XZRect to_xzrect(Material *&mt) {
    float x0, x1, z0, z1, k;
    rect_val(x0, x1, z0, z1, k);
    XZRect xzr(x0, x1, z0, z1, k, mt);
    return xzr;
  }
  __host__ __device__ YZRect to_yzrect(Material *&mt) {
    float y0, y1, z0, z1, k;
    rect_val(y0, y1, z0, z1, k);
    YZRect yzr(y0, y1, z0, z1, k, mt);
    return yzr;
  }
  __host__ __device__ Hittable *to_hittable() {
    MaterialParam mparam = to_material_param();
    Material *mt = mparam.to_material();
    return to_hittable(mt);
  }
  __host__ __device__ Hittable *to_hittable(Material *&mt) {
    Hittable *ht;
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
  to_hittable(unsigned char *&dt) {
    MaterialParam mparam = to_material_param();
    Material *mt = mparam.to_material(dt);
    return to_hittable(mt);
  }
  __device__ Hittable *to_hittable(curandState *loc) {
    MaterialParam mparam = to_material_param();
    Material *mt = mparam.to_material(loc);
    return to_hittable(mt);
  }
  __device__ Hittable *to_hittable(unsigned char *&dt,
                                   curandState *loc) {
    MaterialParam mparam = to_material_param();
    Material *mt = mparam.to_material(dt, loc);
    return to_hittable(mt);
  }
};
#endif
