#ifndef SCENEHIT_CUH
#define SCENEHIT_CUH
// scene hittable
#include <aabb.cuh>
#include <aarect.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <vec3.cuh>

template <class HiT> struct SceneHittable {
  __device__ static bool hit(Hit h, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec);
  __host__ __device__ static bool
  bounding_box(HiT h, float t0, float t1, Aabb &output_box);

  __device__ static float pdf_value(HiT h, const Point3 &o,
                                    const Point3 &v);
  __device__ static Vec3 random(const Vec3 &v,
                                curandState *loc) {
    return Vec3(1.0f, 0.0f, 0.0f);
  }
};
#endif
