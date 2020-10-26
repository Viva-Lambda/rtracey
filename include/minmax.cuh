#pragma once
// compute min max points in primitive and GroupParams.
#include <external.hpp>
#include <groupparam.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <scenetype.cuh>
#include <shapeparam.cuh>
#include <vec3.cuh>

template <HittableType t>
__host__ __device__ Vec3 min_vec(const HittableParam &h) {
  return Vec3(0.0f);
}
template <>
__host__ __device__ Vec3
min_vec<SPHERE>(const HittableParam &p) {
  Point3 center(p.p1x, p.p1y, p.p1z);
  return center - Vec3(p.radius);
}
template <>
__host__ __device__ Vec3
min_vec<MOVING_SPHERE>(const HittableParam &p) {
  Point3 center1(p.p1x, p.p1y, p.p1z);
  Point3 center2(p.p2x, p.p2y, p.p2z);
  Point3 center = (center1 + center2) / 2.0f;
  return center - Vec3(p.radius);
}
template <>
__host__ __device__ Vec3
min_vec<TRIANGLE>(const HittableParam &p) {
  Point3 p1(p.p1x, p.p1y, p.p1z);
  Point3 p2(p.p2x, p.p2y, p.p2z);
  Point3 p3(p.n1x, p.n1y, p.n1z);
  Point3 pmin = min_vec(p1, p2);
  pmin = min_vec(pmin, p3);
  return pmin;
}
template <>
__host__ __device__ Vec3
min_vec<XY_TRIANGLE>(const HittableParam &p) {
  return min_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
min_vec<XZ_TRIANGLE>(const HittableParam &p) {
  return min_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
min_vec<YZ_TRIANGLE>(const HittableParam &p) {
  return min_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
min_vec<RECTANGLE>(const HittableParam &p) {
  float k = p.radius;
  float a0 = p.p1x;
  float b0 = p.p2x;
  Vec3 anormal = Vec3(p.n1x, p.n1y, p.n1z);
  AxisInfo ax = AxisInfo(anormal);

  Point3 p1;
  // choose points with axis
  switch (ax.notAligned) {
  case 2: {
    p1 = Point3(a0, b0, k - 0.0001);
    break;
  }
  case 1: {
    p1 = Point3(a0, k - 0.0001, b0);
    break;
  }
  case 0: {
    p1 = Point3(k - 0.0001, a0, b0);
    break;
  }
  }
  return p1;
}

template <>
__host__ __device__ Vec3
min_vec<XY_RECT>(const HittableParam &p) {
  //
  return min_vec<RECTANGLE>(p);
}
template <>
__host__ __device__ Vec3
min_vec<XZ_RECT>(const HittableParam &p) {
  return min_vec<RECTANGLE>(p);
}
template <>
__host__ __device__ Vec3
min_vec<YZ_RECT>(const HittableParam &p) {
  return min_vec<RECTANGLE>(p);
}

template <>
__host__ __device__ Vec3
min_vec<HITTABLE>(const HittableParam &p) {
  HittableType htype = static_cast<HittableType>(p.htype);
  Vec3 res(0.0f);
  if (htype == NONE_HITTABLE) {
    return res;
  } else if (htype == SPHERE) {
    res = min_vec<SPHERE>(p);
  } else if (htype == MOVING_SPHERE) {
    res = min_vec<MOVING_SPHERE>(p);
  } else if (htype == TRIANGLE) {
    res = min_vec<TRIANGLE>(p);
  } else if (htype == RECTANGLE) {
    res = min_vec<RECTANGLE>(p);
  } else if (htype == XY_RECT) {
    res = min_vec<XY_RECT>(p);
  } else if (htype == XZ_RECT) {
    res = min_vec<XZ_RECT>(p);
  } else if (htype == YZ_RECT) {
    res = min_vec<YZ_RECT>(p);
  } else if (htype == YZ_TRIANGLE) {
    res = min_vec<YZ_TRIANGLE>(p);
  } else if (htype == XZ_TRIANGLE) {
    res = min_vec<XZ_TRIANGLE>(p);
  } else if (htype == XY_TRIANGLE) {
    res = min_vec<XY_TRIANGLE>(p);
  }
  return res;
}
template <HittableType t>
__host__ __device__ Vec3 max_vec(const HittableParam &h) {
  return Vec3(0.0f);
}
template <>
__host__ __device__ Vec3
max_vec<SPHERE>(const HittableParam &p) {
  Point3 center(p.p1x, p.p1y, p.p1z);
  return center + Vec3(p.radius);
}
template <>
__host__ __device__ Vec3
max_vec<MOVING_SPHERE>(const HittableParam &p) {
  Point3 center1(p.p1x, p.p1y, p.p1z);
  Point3 center2(p.p2x, p.p2y, p.p2z);
  Point3 center = (center1 + center2) / 2.0f;
  return center + Vec3(p.radius);
}

template <>
__host__ __device__ Vec3
max_vec<TRIANGLE>(const HittableParam &p) {
  Point3 p1(p.p1x, p.p1y, p.p1z);
  Point3 p2(p.p2x, p.p2y, p.p2z);
  Point3 p3(p.n1x, p.n1y, p.n1z);
  Point3 pmin = max_vec(p1, p2);
  pmin = max_vec(pmin, p3);
  return pmin;
}
template <>
__host__ __device__ Vec3
max_vec<XY_TRIANGLE>(const HittableParam &p) {
  return max_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<XZ_TRIANGLE>(const HittableParam &p) {
  return max_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<YZ_TRIANGLE>(const HittableParam &p) {
  return max_vec<TRIANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<RECTANGLE>(const HittableParam &p) {
  float k = p.radius;
  float a1 = p.p1y;
  float b1 = p.p2y;
  Vec3 anormal = Vec3(p.n1x, p.n1y, p.n1z);
  AxisInfo ax = AxisInfo(anormal);

  Point3 p2;
  // choose points with axis
  switch (ax.notAligned) {
  case 2:
    p2 = Point3(a1, b1, k + 0.0001);
    break;
  case 1:
    p2 = Point3(a1, k + 0.0001, b1);
    break;
  case 0:
    p2 = Point3(k + 0.0001, a1, b1);
    break;
  }

  return p2;
}
template <>
__host__ __device__ Vec3
max_vec<XY_RECT>(const HittableParam &p) {
  //
  return max_vec<RECTANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<XZ_RECT>(const HittableParam &p) {
  return max_vec<RECTANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<YZ_RECT>(const HittableParam &p) {
  return max_vec<RECTANGLE>(p);
}
template <>
__host__ __device__ Vec3
max_vec<HITTABLE>(const HittableParam &p) {
  //
  HittableType htype = static_cast<HittableType>(p.htype);
  Vec3 res(0.0f);
  if (htype == NONE_HITTABLE) {
    return res;
  } else if (htype == SPHERE) {
    res = max_vec<SPHERE>(p);
  } else if (htype == MOVING_SPHERE) {
    res = max_vec<MOVING_SPHERE>(p);
  } else if (htype == TRIANGLE) {
    res = max_vec<TRIANGLE>(p);
  } else if (htype == RECTANGLE) {
    res = max_vec<RECTANGLE>(p);
  } else if (htype == XY_RECT) {
    res = max_vec<XY_RECT>(p);
  } else if (htype == XZ_RECT) {
    res = max_vec<XZ_RECT>(p);
  } else if (htype == YZ_RECT) {
    res = max_vec<YZ_RECT>(p);
  } else if (htype == YZ_TRIANGLE) {
    res = max_vec<YZ_TRIANGLE>(p);
  } else if (htype == XZ_TRIANGLE) {
    res = max_vec<XZ_TRIANGLE>(p);
  } else if (htype == XY_TRIANGLE) {
    res = max_vec<XY_TRIANGLE>(p);
  }
  return res;
}
template <typename T>
__host__ __device__ Vec3 min_vec(const T &p) {
  return Vec3(0.0f);
}
template <typename T>
__host__ __device__ Vec3 max_vec(const T &p) {
  return Vec3(0.0f);
}
