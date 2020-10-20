#pragma once
#include <matparam.cuh>
#include <ray.cuh>
#include <shape.cuh>
#include <shapeparam.cuh>
#include <vec3.cuh>

struct Primitive {
  // material params
  MaterialParam mparam;

  // hittable params
  HittableType htype; //

  float p1x, p1y, p1z;
  float p2x, p2y, p2z;
  float radius;
  float n1x, n1y, n1z;

  // group params
  int group_id;
  int group_index;

  __host__ __device__ Primitive()
      : group_id(0), group_index(0), p1x(0.0f), p1y(0.0f),
        p1z(0.0f), p2x(0.0f), p2y(0.0f), p2z(0.0f),
        n1x(0.0f), n1y(0.0f), n1z(0.0f), radius(0.0f),
        htype(NONE_HITTABLE) {}
  __host__ __device__ Primitive(const MaterialParam &mt,
                                const HittableParam &ht,
                                const int gindex,
                                const int gid)
      : mparam(mt), group_index(gindex), group_id(gid),
        htype(ht.htype), p1x(ht.p1x), p1y(ht.p1y),
        p1z(ht.p1z), p2x(ht.p2x), p2y(ht.p2y), p2z(ht.p2z),
        n1x(ht.n1x), n1y(ht.n1y), n1z(ht.n1z),
        radius(ht.radius) {}
  __host__ __device__ Primitive(const Primitive &p)
      : mparam(p.mparam), htype(p.htype), p1x(p.p1x),
        p1y(p.p1y), p1z(p.p1z), p2x(p.p2x), p2y(p.p2y),
        p2z(p.p2z), n1x(p.n1x), n1y(p.n1y), n1z(p.n1z),
        radius(p.radius), group_id(p.group_id),
        group_index(p.group_index) {}
  __host__ __device__ Primitive &
  operator=(const Primitive &p) {
    //
    mparam = p.mparam;
    htype = p.htype;

    p1x = p.p1x;
    p1y = p.p1y;
    p1z = p.p1z;
    //
    p2x = p.p2x;
    p2y = p.p2y;
    p2z = p.p2z;
    //
    n1x = p.n1x;
    n1y = p.n1y;
    n1z = p.n1z;
    //
    radius = p.radius;
    group_id = p.group_id;
    group_index = p.group_index;
    return *this;
  }
};
