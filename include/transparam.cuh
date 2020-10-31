#pragma once
#include <scenetype.cuh>
// transformation parameters
struct TransParam {
  TransformationType transtype;
  Point3 displacement;
  float degree;
  Vec3 axis;
  float scale;
  __host__ __device__ TransParam() {}
};
__host__ __device__ TransParam mkTranslate(Point3 steps) {
  TransParam tp;
  tp.transtype = TRANSLATE;
  tp.displacement = steps;
  return tp;
}
__host__ __device__ TransParam mkRotate(float degree,
                                        Vec3 ax) {
  TransParam tp;
  tp.transtype = ROTATE;
  tp.degree = degree;
  tp.axis = ax;
  return tp;
}
__host__ __device__ TransParam mkRotateY(float degree) {
  return mkRotate(degree, Vec3(0.0f, 1.0f, 0.0f));
}
__host__ __device__ TransParam mkRotateZ(float degree) {
  return mkRotate(degree, Vec3(0.0f, 0.0f, 1.0f));
}
__host__ __device__ TransParam mkRotateX(float degree) {
  return mkRotate(degree, Vec3(1.0f, 0.0f, 0.0f));
}
__host__ __device__ TransParam mkRotateTranslate(
    float degree, const Vec3 &ax, const Point3 &step) {
  TransParam tp;
  tp.transtype = ROTATE_TRANSLATE;
  tp.degree = degree;
  tp.axis = ax;
  tp.displacement = step;
  return tp;
}
__host__ __device__ TransParam mkScale(float s) {
  TransParam tp;
  tp.scale = s;
  return tp;
}
__host__ __device__ TransParam
mkScaleTranslate(float s, const Point3 &step) {
  TransParam tp;
  tp.scale = s;
  tp.displacement = step;
  return tp;
}
__host__ __device__ TransParam mkScaleRotate(float s,
                                             const Vec3 &ax,
                                             float deg) {
  TransParam tp;
  tp.scale = s;
  tp.axis = ax;
  tp.degree = deg;
  return tp;
}
__host__ __device__ TransParam
mkScaleRotateTranslate(float s, const Vec3 &ax, float deg,
                       const Point3 &step) {
  TransParam tp;
  tp.scale = s;
  tp.axis = ax;
  tp.degree = deg;
  tp.displacement = step;
  return tp;
}
