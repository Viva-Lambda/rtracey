#pragma once

#include <ray.cuh>
#include <vec3.cuh>

struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  float u, v;
  bool front_face;

  // scene related params
  int group_id;
  int group_index;
  int primitive_index;
  bool is_group_scattering = false;

  __host__ __device__ HitRecord()
      : t(0.0f), p(Vec3(0.0f)), u(0.0f), v(0.0f),
        front_face(false), group_id(0), group_index(0),
        primitive_index(0) {}

  /**
    @brief check if ray hits the front side of the object

    We check the angle between the incoming ray and
    hit point. When vectors point in same direction (not
    necessarily parallel)
    their dot product is positive.

    @param r incoming ray
    @param norm the surface normal of the hit point.
   */
  __host__ __device__ void
  set_front_face(const Ray &r, const Vec3 &norm) {
    front_face = dot(r.direction(), norm) < 0.0f;
    normal = front_face ? norm : -norm;
  }
};

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
