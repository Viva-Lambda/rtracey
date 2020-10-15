#pragma once
#include <matparam.cuh>
#include <ray.cuh>
#include <shape.cuh>
#include <shapeparam.cuh>
#include <vec3.cuh>

struct Primitive {
  // material params
  MaterialParam mparam;

  // hittable params
  HittableParam hparam;

  // group params
  int group_id;
  int group_index;

  __host__ __device__ Primitive() {}
  __host__ __device__ Primitive(MaterialParam mt,
                                HittableParam ht,
                                int gindex, int gid)
      : mparam(mt), hparam(ht), group_index(gindex),
        group_id(gid) {}
  __host__ __device__ Sphere to_sphere() const {
    Point3 cent(hparam.p1x, hparam.p1y, hparam.p1z);
    Sphere sp(cent, hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ MovingSphere
  to_moving_sphere() const {
    Point3 cent1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 cent2(hparam.p2x, hparam.p2y, hparam.p2z);
    MovingSphere sp(cent1, cent2, hparam.n1x, hparam.n1y,
                    hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ Triangle to_triangle() const {
    Point3 p1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 p2(hparam.p2x, hparam.p2y, hparam.p2z);
    Point3 p3(hparam.n1x, hparam.n1y, hparam.n1z);
    Triangle tri(p1, p2, p3, mparam);
    return tri;
  }
  __host__ __device__ void rect_val(float &a0, float &a1,
                                    float &b0, float &b1,
                                    float &k) const {
    a0 = hparam.p1x;
    a1 = hparam.p1y;
    b0 = hparam.p2x;
    b1 = hparam.p2y;
    k = hparam.radius;
  }
  __host__ __device__ XYRect to_xyrect() const {
    float x0, x1, y0, y1, k;
    rect_val(x0, x1, y0, y1, k);
    XYRect xyr(x0, x1, y0, y1, k, mparam);
    return xyr;
  }
  __host__ __device__ XZRect to_xzrect() const {
    float x0, x1, z0, z1, k;
    rect_val(x0, x1, z0, z1, k);
    XZRect xzr(x0, x1, z0, z1, k, mparam);
    return xzr;
  }
  __host__ __device__ YZRect to_yzrect() const {
    float y0, y1, z0, z1, k;
    rect_val(y0, y1, z0, z1, k);
    YZRect yzr(y0, y1, z0, z1, k, mparam);
    return yzr;
  }
};