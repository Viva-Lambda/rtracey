#pragma once
#include <ray.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  MaterialParam mat_ptr;
  float u, v;
  bool front_face;

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
