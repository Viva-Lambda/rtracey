#pragma once
#include <vec3.cuh>

struct AxisInfo {
  int aligned1;
  int aligned2;
  int notAligned;
  __host__ __device__ AxisInfo() {}
  __host__ __device__ AxisInfo(Vec3 anormal) {
    if (anormal.z() == 1.0) {
      aligned1 = 0;
      aligned2 = 1;
      notAligned = 2;
    } else if (anormal.x() == 1.0) {
      aligned1 = 1;
      aligned2 = 2;
      notAligned = 0;
    } else if (anormal.y() == 1.0) {
      aligned1 = 0;
      aligned2 = 2;
      notAligned = 1;
    }
  }
};
