#pragma once
#include <scenetype.cuh>
#include <vec3.cuh>
#include <matrix.cuh>

struct HittableParam {
  const HittableType htype; //

  const float p1x, p1y, p1z;
  const float p2x, p2y, p2z;
  const float radius;
  const float n1x, n1y, n1z;
  __host__ __device__ HittableParam()
      : p1x(0.0f), p1y(0.0f), p1z(0.0f), p2x(0.0f),
        p2y(0.0f), p2z(0.0f), n1x(0.0f), n1y(0.0f),
        n1z(0.0f), radius(0.0f), htype(NONE_HITTABLE) {}
  __host__ __device__ HittableParam(
      const HittableType ht, const float _p1x,
      const float _p1y, const float _p1z, const float _p2x,
      const float _p2y, const float _p2z, const float _n1x,
      const float _n1y, const float _n1z, const float r)
      : htype(ht), p1x(_p1x), p1y(_p1y), p1z(_p1z),
        p2x(_p2x), p2y(_p2y), p2z(_p2z), n1x(_n1x),
        n1y(_n1y), n1z(_n1z), radius(r) {}
};

__host__ __device__ HittableParam mkRectHittable(
    const float a0, const float a1, const float b0,
    const float b1, Vec3 anormal, const float k) {

  HittableType htype;
  if (anormal.z() == 1) {
    htype = XY_RECT;
  } else if (anormal.y() == 1) {
    htype = XZ_RECT;
  } else if (anormal.x() == 1) {
    htype = YZ_RECT;
  } else {
    htype = NONE_HITTABLE;
  }
  float nx = anormal.x();
  float ny = anormal.y();
  float nz = anormal.z();
  HittableParam param(htype, a0, a1, 0, b0, b1, 0, nx, ny,
                      nz, k);
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
  HittableParam param(SPHERE, cent.x(), cent.y(), cent.z(),
                      0, 0, 0, 0, 0, 0, rad);
  return param;
}
__host__ __device__ HittableParam
mkMovingSphereHittable(Point3 cent1, Point3 cent2,
                       float rad, float t0, float t1) {
  HittableParam param(MOVING_SPHERE, cent1.x(), cent1.y(),
                      cent1.z(), cent2.x(), cent2.y(),
                      cent2.z(), t0, t1, 0, rad);
  return param;
}

__host__ __device__ HittableParam translate(const HittableParam& hparam,

