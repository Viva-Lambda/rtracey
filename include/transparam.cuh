#pragma once
#include <scenetype.cuh>
// transformation parameters
struct TransParam {
  TransformationType transtype;
  Point3 displacement;
  float degree;
  __host__ __device__ TransParam() {}
};
__host__ __device__ TransParam mkTranslation(Point3 steps) {
  TransParam tp;
  tp.transtype = TRANSLATE;
  tp.displacement = steps;
  return tp;
}
__host__ __device__ TransParam mkRotateY(float degree) {
  TransParam tp;
  tp.transtype = ROTATE_Y;
  tp.degree = degree;
  return tp;
}
