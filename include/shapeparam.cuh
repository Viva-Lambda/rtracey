#pragma once
#include <scenetype.cuh>
#include <vec3.cuh>

struct HittableParam {
  const int *htype; //

  const float *p1x, *p1y, *p1z;
  const float *p2x, *p2y, *p2z;
  const float *radius;
  const float *n1x, *n1y, *n1z;
  __host__ __device__ HittableParam()
      : p1x(nullptr), p1y(nullptr), p1z(nullptr),
        p2x(nullptr), p2y(nullptr), p2z(nullptr),
        n1x(nullptr), n1y(nullptr), n1z(nullptr),
        radius(nullptr), htype(nullptr) {}
  __host__ __device__ HittableParam(
      const int *ht, const float *_p1x, const float *_p1y,
      const float *_p1z, const float *_p2x,
      const float *_p2y, const float *_p2z,
      const float *_n1x, const float *_n1y,
      const float *_n1z, const float *r)
      : htype(ht), p1x(_p1x), p1y(_p1y), p1z(_p1z),
        p2x(_p2x), p2y(_p2y), p2z(_p2z), n1x(_n1x),
        n1y(_n1y), n1z(_n1z), radius(r) {}
};

__host__ __device__ HittableParam mkRectHittable(
    const float a0, const float a1, const float b0,
    const float b1, Vec3 anormal, const float k) {

  const int *htype;
  if (anormal.z() == 1) {
    htype = &XY_RECT;
  } else if (anormal.y() == 1) {
    htype = &XZ_RECT;
  } else if (anormal.x() == 1) {
    htype = &YZ_RECT;
  } else {
    htype = &NONE_HITTABLE;
  }
  HittableParam param(
      htype, &a0, &a1, (const float *)nullptr, &b0, &b1,
      (const float *)nullptr, (const float *)nullptr,
      (const float *)nullptr, (const float *)nullptr, &k);
  return param;
}
__host__ __device__ HittableParam mkYZRectHittable(
    float a0, float a1, float b0, float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(1, 0, 0), k);
}
__host__ __device__ HittableParam mkXZRectHittable(
    float a0, float a1, float b0, float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(0, 1, 0), k);
}
__host__ __device__ HittableParam mkXYRectHittable(
    float a0, float a1, float b0, float b1, float k) {
  return mkRectHittable(a0, a1, b0, b1, Vec3(0, 0, 1), k);
}
HittableParam mkSphereHittable(Point3 cent, float rad) {
  HittableParam param(&SPHERE, cent.e1, cent.e2, cent.e3,
                      nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, &rad);
  return param;
}
__host__ __device__ HittableParam
mkMovingSphereHittable(Point3 cent1, Point3 cent2,
                       float rad, float t0, float t1) {
  HittableParam param(&MOVING_SPHERE, cent1.e1, cent1.e2,
                      cent1.e3, cent2.e1, cent2.e2,
                      cent2.e3, &t0, &t1, nullptr, &rad);
  return param;
}
