#pragma once
#include <matparam.cuh>
#include <ray.cuh>
#include <shape.cuh>
#include <shapeparam.cuh>
#include <vec3.cuh>

struct Primitive {
  // material params
  const MaterialParam mparam;

  // hittable params
  const HittableParam hparam;

  // group params
  const int group_id;
  const int group_index;

  __host__ __device__ Primitive()
      : group_id(0), group_index(0) {}
  __host__ __device__ Primitive(const MaterialParam &mt,
                                const HittableParam &ht,
                                const int gindex,
                                const int gid)
      : mparam(mt), hparam(ht), group_index(gindex),
        group_id(gid) {}
  __host__ __device__ Primitive(const Primitive &p)
      : mparam(p.mparam), hparam(p.hparam),
        group_id(p.group_id), group_index(p.group_index) {}
};
