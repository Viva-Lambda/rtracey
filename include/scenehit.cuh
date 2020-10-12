#ifndef SCENEHIT_CUH
#define SCENEHIT_CUH
// scene hittable
#include <aabb.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct Hittable {
  MaterialParam mat_ptr;
  __device__ virtual bool hit(const Ray &r, float d_min,
                              float d_max,
                              HitRecord &rec) const = 0;
  __host__ __device__ virtual bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const = 0;
  __device__ virtual float
  pdf_value(const Point3 &o, const Point3 &v) const {
    return 0.0f;
  }
  __device__ virtual Vec3 random(const Vec3 &v,
                                 curandState *loc) const {
    return Vec3(1.0f, 0.0f, 0.0f);
  }
};
template <class HiT> struct SceneHittable {

  template <typename T>
  __device__ static bool hit(const HiT &h, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec);
  template <typename T>
  __host__ __device__ static bool
  bounding_box(const HiT &h, float t0, float t1,
               Aabb &output_box);

  template <typename T>
  __device__ static float pdf_value(const HiT &&h,
                                    const Point3 &o,
                                    const Point3 &v);
  template <typename T>
  __device__ static Vec3 random(const HiT &h, const Vec3 &v,
                                curandState *loc) {
    return Vec3(1.0f, 0.0f, 0.0f);
  }
};
#endif
