#pragma once
#include <matparam.cuh>
#include <ray.cuh>
#include <shape.cuh>
#include <shapeparam.cuh>
#include <vec3.cuh>

struct Primitive {
  // material params
  const MaterialParam mparam;

  // hittable params
  const HittableParam hparam;

  // group params
  const int *group_id;
  const int *group_index;

  __host__ __device__ Primitive()
      : group_id(nullptr), group_index(nullptr) {}
  __host__ __device__ Primitive(const MaterialParam &mt,
                                const HittableParam &ht,
                                const int *gindex,
                                const int *gid)
      : mparam(mt), hparam(ht), group_index(gindex),
        group_id(gid) {}
  __host__ __device__ Primitive(const Primitive &p)
      : mparam(p.mparam), hparam(p.hparam),
        group_id(p.group_id), group_index(p.group_index) {}

  __host__ __device__ Sphere to_sphere() const {
    Point3 cent((float *)hparam.p1x, (float *)hparam.p1y,
                (float *)hparam.p1z);
    Sphere sp(cent, (float *)hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ MovingSphere
  to_moving_sphere() const {
    Point3 cent1((float *)hparam.p1x, (float *)hparam.p1y,
                 (float *)hparam.p1z);
    Point3 cent2((float *)hparam.p2x, (float *)hparam.p2y,
                 (float *)hparam.p2z);
    MovingSphere sp(cent1, cent2, (float *)hparam.n1x,
                    (float *)hparam.n1y,
                    (float *)hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ Triangle to_triangle() const {
    Point3 p1((float *)hparam.p1x, (float *)hparam.p1y,
              (float *)hparam.p1z);
    Point3 p2((float *)hparam.p2x, (float *)hparam.p2y,
              (float *)hparam.p2z);
    Point3 p3((float *)hparam.n1x, (float *)hparam.n1y,
              (float *)hparam.n1z);
    Triangle tri(p1, p2, p3, mparam);
    return tri;
  }
  __host__ __device__ void rect_val(float *&a0, float *&a1,
                                    float *&b0, float *&b1,
                                    float *&k) const {
    a0 = (float *)hparam.p1x;
    a1 = (float *)hparam.p1y;
    b0 = (float *)hparam.p2x;
    b1 = (float *)hparam.p2y;
    k = (float *)hparam.radius;
  }
  __host__ __device__ XYRect to_xyrect() const {
    float *x0, *x1, *y0, *y1, *k;
    rect_val(x0, x1, y0, y1, k);
    XYRect xyr(x0, x1, y0, y1, k, mparam);
    return xyr;
  }
  __host__ __device__ XZRect to_xzrect() const {
    float *x0, *x1, *z0, *z1, *k;
    rect_val(x0, x1, z0, z1, k);
    XZRect xzr(x0, x1, z0, z1, k, mparam);
    return xzr;
  }
  __host__ __device__ YZRect to_yzrect() const {
    float *y0, *y1, *z0, *z1, *k;
    rect_val(y0, y1, z0, z1, k);
    YZRect yzr(y0, y1, z0, z1, k, mparam);
    return yzr;
  }
};
