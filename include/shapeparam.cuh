#pragma once
#include <matrix.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

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
  __host__ __device__ HittableParam(const HittableParam &hp)
      : htype(hp.htype), p1x(hp.p1x), p1y(hp.p1y),
        p1z(hp.p1z), p2x(hp.p2x), p2y(hp.p2y), p2z(hp.p2z),
        n1x(hp.n1x), n1y(hp.n1y), n1z(hp.n1z),
        radius(hp.radius) {}
  __host__ __device__ HittableParam(const HittableType &ht,
                                    const Point3 &p1,
                                    const Point3 &p2,
                                    const Vec3 &n1,
                                    const float r)
      : htype(ht), p1x(p1.x()), p1y(p1.y()), p1z(p1.z()),
        p2x(p2.x()), p2y(p2.y()), p2z(p2.z()), n1x(n1.x()),
        n1y(n1.y()), n1z(n1.z()), radius(r) {}

  __host__ __device__ HittableParam(
      const HittableType ht, const float _p1x,
      const float _p1y, const float _p1z, const float _p2x,
      const float _p2y, const float _p2z, const float _n1x,
      const float _n1y, const float _n1z, const float r)
      : htype(ht), p1x(_p1x), p1y(_p1y), p1z(_p1z),
        p2x(_p2x), p2y(_p2y), p2z(_p2z), n1x(_n1x),
        n1y(_n1y), n1z(_n1z), radius(r) {}
  __host__ __device__ Point3 get_point1() const {
    return Point3(p1x, p1y, p1z);
  }
  __host__ __device__ Point3 get_point2() const {
    return Point3(p2x, p2y, p2z);
  }
  __host__ __device__ Vec3 get_normal() const {
    return Point3(n1x, n1y, n1z);
  }
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
  HittableParam param(htype, a0, b0, k, a1, b1, k, nx, ny,
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
__host__ __device__ HittableParam
mkSphereHittable(Point3 cent, float rad) {
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

__host__ __device__ HittableParam
translate(const HittableParam &hparam, Point3 steps) {
  Point3 p1(hparam.p1x, hparam.p1y, hparam.p1z);
  Point3 p2(hparam.p2x, hparam.p2y, hparam.p2z);
  Matrix transMat =
      translate(steps.x(), steps.y(), steps.z());
  Point3 np1 = transMat * p1;
  Point3 np2 = transMat * p2;
  HittableParam hp(hparam.htype, np1, np2,
                   hparam.get_normal(), hparam.radius);
  return hp;
}
__host__ __device__ HittableParam rotate(
    const HittableParam &hparam, Vec3 axis, float degree) {
  float radian = degree_to_radian(degree);
  Point3 p1 = hparam.get_point1();
  Point3 p2 = hparam.get_point2();
  Vec3 n1 = hparam.get_normal();
  Matrix rotMat = rotate(axis, radian);
  Point3 np1 = rotMat * p1;
  Point3 np2 = rotMat * p2;
  Point3 nn1 = rotMat * n1;
  HittableParam hp(hparam.htype, np1, np2, nn1,
                   hparam.radius);
  return hp;
}
__host__ __device__ HittableParam
rotate_y(const HittableParam &hparam, float degree) {
  float radian = degree_to_radian(degree);
  Point3 p1 = hparam.get_point1();
  Point3 p2 = hparam.get_point2();
  Vec3 n1 = hparam.get_normal();

  Matrix rotMat = rotateY(radian);
  Point3 np1 = rotMat * p1;
  Point3 np2 = rotMat * p2;
  Point3 nn1 = rotMat * n1;
  HittableParam hp(hparam.htype, np1, np2, nn1,
                   hparam.radius);
  return hp;
}
__host__ __device__ HittableParam
rotate_x(const HittableParam &hparam, float degree) {
  float radian = degree_to_radian(degree);
  Point3 p1 = hparam.get_point1();
  Point3 p2 = hparam.get_point2();
  Vec3 n1 = hparam.get_normal();

  Matrix rotMat = rotateX(radian);
  Point3 np1 = rotMat * p1;
  Point3 np2 = rotMat * p2;
  Point3 nn1 = rotMat * n1;
  HittableParam hp(hparam.htype, np1, np2, nn1,
                   hparam.radius);
  return hp;
}
__host__ __device__ HittableParam
rotate_z(const HittableParam &hparam, float degree) {
  float radian = degree_to_radian(degree);
  Point3 p1 = hparam.get_point1();
  Point3 p2 = hparam.get_point2();
  Vec3 n1 = hparam.get_normal();

  Matrix rotMat = rotateZ(radian);
  Point3 np1 = rotMat * p1;
  Point3 np2 = rotMat * p2;
  Point3 nn1 = rotMat * n1;
  HittableParam hp(hparam.htype, np1, np2, nn1,
                   hparam.radius);
  return hp;
}
