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
  // material params
  MaterialParam mparam;

  // hittable params
  HittableParam hparam;

  // group params
  int group_id;
  int group_index;

  __host__ __device__ ScenePrim() {}
  __host__ __device__ ScenePrim(MaterialParam mt,
                                HittableParam ht,
                                int gindex, int gid)
      : mparam(mt), hparam(ht), group_index(gindex),
        group_id(gid) {}

  __host__ __device__ Sphere to_sphere(Material *&mt) {
    Point3 cent(hparam.p1x, hparam.p1y, hparam.p1z);
    Sphere sp(cent, hparam.radius, mt);
    return sp;
  }
  __host__ __device__ MovingSphere
  to_moving_sphere(Material *&mt) {
    Point3 cent1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 cent2(hparam.p2x, hparam.p2y, hparam.p2z);
    MovingSphere sp(cent1, cent2, hparam.n1x, hparam.n1y,
                    hparam.radius, mt);
    return sp;
  }
  __host__ __device__ Triangle to_triangle(Material *&mt) {
    Point3 p1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 p2(hparam.p2x, hparam.p2y, hparam.p2z);
    Point3 p3(hparam.n1x, hparam.n1y, hparam.n1z);
    Triangle tri(p1, p2, p3, mt);
    return tri;
  }
  __host__ __device__ void rect_val(float &a0, float &a1,
                                    float &b0, float &b1,
                                    float &k) {
    a0 = hparam.p1x;
    a1 = hparam.p1y;
    b0 = hparam.p2x;
    b1 = hparam.p2y;
    k = hparam.radius;
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
    Material *mt = mparam.to_material();
    return to_hittable(mt);
  }
  __host__ __device__ Hittable *to_hittable(Material *&mt) {
    Hittable *ht;
    switch (hparam.htype) {
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
    Material *mt = mparam.to_material(dt);
    return to_hittable(mt);
  }
  __device__ Hittable *to_hittable(curandState *loc) {
    Material *mt = mparam.to_material(loc);
    return to_hittable(mt);
  }
  __device__ Hittable *to_hittable(unsigned char *&dt,
                                   curandState *loc) {
    Material *mt = mparam.to_material(dt, loc);
    return to_hittable(mt);
  }
};
#endif
