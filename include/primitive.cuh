#pragma once
#include <matparam.cuh>
#include <minmax.cuh>
#include <ray.cuh>
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
  __host__ __device__ HittableParam get_hparam() const {
    HittableParam hp(htype, p1x, p1y, p1z, p2x, p2y, p2z,
                     n1x, n1y, n1z, radius);
    return hp;
  }
  __host__ __device__ void set_hparam(HittableParam hp) {
    htype = hp.htype;

    p1x = hp.p1x;
    p1y = hp.p1y;
    p1z = hp.p1z;

    p2x = hp.p2x;
    p2y = hp.p2y;
    p2z = hp.p2z;

    n1x = hp.n1x;
    n1y = hp.n1y;
    n1z = hp.n1z;
    radius = hp.radius;
  }
};

template <>
__host__ __device__ Vec3
min_vec<Primitive>(const Primitive &p) {
  HittableParam h = p.get_hparam();
  return min_vec<HITTABLE>(h);
}
template <>
__host__ __device__ Vec3
max_vec<Primitive>(const Primitive &p) {
  HittableParam h = p.get_hparam();
  return max_vec<HITTABLE>(h);
}

__host__ __device__ Primitive translate(Primitive &p,
                                        Point3 steps) {
  HittableParam hp = translate(p.get_hparam(), steps);
  Primitive pr(p.mparam, hp, p.group_index, p.group_id);
  return pr;
}
__host__ __device__ Primitive rotate(Primitive &p,
                                     Vec3 axis,
                                     float degree) {
  HittableParam hp = rotate(p.get_hparam(), axis, degree);
  Primitive pr(p.mparam, hp, p.group_index, p.group_id);
  return pr;
}
__host__ __device__ Primitive rotate_y(Primitive &p,
                                       float degree) {
  HittableParam hp = rotate_y(p.get_hparam(), degree);
  Primitive pr(p.mparam, hp, p.group_index, p.group_id);
  return pr;
}
__host__ __device__ Primitive rotate_x(Primitive &p,
                                       float degree) {
  HittableParam hp = rotate_x(p.get_hparam(), degree);
  Primitive pr(p.mparam, hp, p.group_index, p.group_id);
  return pr;
}
__host__ __device__ Primitive rotate_z(Primitive &p,
                                       float degree) {
  HittableParam hp = rotate_z(p.get_hparam(), degree);
  Primitive pr(p.mparam, hp, p.group_index, p.group_id);
  return pr;
}
