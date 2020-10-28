#pragma once
#include <group.cuh>
#include <minmax.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

struct GroupParam {
  // group params
  GroupType gtype;
  int group_size;
  int group_id;
  Primitive *prims;
  //
  float density;

  TextureType ttype;
  float tp1x, tp1y, tp1z;
  float scale;
  int width, height, bytes_per_pixel, index;

  MaterialType mtype;
  float fuzz_ref_idx;

  float minx, miny, minz;
  float maxx, maxy, maxz;

  __host__ __device__ GroupParam()
      : gtype(NONE_GRP), group_size(0), group_id(0),
        density(0.0f), prims(nullptr), width(0), height(0),
        bytes_per_pixel(0), index(0), tp1x(0), tp1y(0),
        tp1z(0), scale(0), ttype(NONE_TEXTURE),
        mtype(NONE_MATERIAL), fuzz_ref_idx(0.0f),
        minx(0.0f), miny(0.0f), minz(0.0f), maxx(0.0f),
        maxy(0.0f), maxz(0.0f) {}
  __host__ __device__ GroupParam(Primitive *prm,
                                 const int gsize,
                                 const int gid,
                                 const GroupType gtp,
                                 const float d,
                                 const MaterialParam &mp)
      : group_size(gsize), group_id(gid), gtype(gtp),
        density(d), width(mp.tparam.width),
        height(mp.tparam.height), mtype(mp.mtype),
        fuzz_ref_idx(mp.fuzz_ref_idx),
        bytes_per_pixel(mp.tparam.bytes_per_pixel),
        index(mp.tparam.index), tp1x(mp.tparam.tp1x),
        tp1y(mp.tparam.tp1y), tp1z(mp.tparam.tp1z),
        scale(mp.tparam.scale), ttype(mp.tparam.ttype) {
    deepcopy(prims, prm, group_size);
    update_max_vec();
    update_min_vec();
  }
  __host__ __device__ void update_max_vec() {
    Vec3 maxv(FLT_MIN);
    for (int i = 0; i < group_size; i++) {
      Primitive p = prims[i];
      Vec3 mxvec = max_vec<Primitive>(p);
      for (int j = 0; j < 3; j++) {
        if (mxvec[j] > maxv[j]) {
          maxv[j] = mxvec[j];
        }
      }
    }
    maxx = maxv.x();
    maxy = maxv.y();
    maxz = maxv.z();
  }
  __host__ __device__ void update_min_vec() {
    Vec3 minv(FLT_MAX);
    for (int i = 0; i < group_size; i++) {
      Primitive p = prims[i];
      Vec3 mnvec = min_vec<Primitive>(p);
      for (int j = 0; j < 3; j++) {
        if (mnvec[j] < minv[j]) {
          minv[j] = mnvec[j];
        }
      }
    }
    minx = minv.x();
    miny = minv.y();
    minz = minv.z();
  }
  __host__ __device__ void update_minmax() {
    update_max_vec();
    update_min_vec();
  }
  __host__ __device__ MaterialParam get_mparam() const {
    TextureParam tp = get_tparam();
    MaterialParam mp(tp, mtype, fuzz_ref_idx);
    return mp;
  }
  __host__ __device__ TextureParam get_tparam() const {
    TextureParam tp(ttype, tp1x, tp1y, tp1z, scale, width,
                    height, bytes_per_pixel, index);
    return tp;
  }
  __host__ __device__ GroupParam(const GroupParam &g)
      : group_size(g.group_size), group_id(g.group_id),
        gtype(g.gtype), density(g.density), width(g.width),
        height(g.height), index(g.index),
        bytes_per_pixel(g.bytes_per_pixel), tp1x(g.tp1x),
        tp1y(g.tp1y), tp1z(g.tp1z), scale(g.scale),
        ttype(g.ttype), mtype(g.mtype), minx(g.minx),
        miny(g.miny), minz(g.minz), maxx(g.maxx),
        maxy(g.maxy), maxz(g.maxz),
        fuzz_ref_idx(g.fuzz_ref_idx) {
    Primitive *prm = g.prims;
    deepcopy(prims, prm, group_size);
  }
  __host__ __device__ void g_free() { delete[] prims; }
  __host__ __device__ Primitive get(int i) const {
    if (i <= 0) {
      return prims[0];
    } else if (i < group_size) {
      return prims[i];
    } else {
      return prims[group_size - 1];
    }
  }
  __host__ __device__ GroupParam &
  operator=(const GroupParam &g) {
    gtype = g.gtype;
    group_size = g.group_size;
    group_id = g.group_id;

    deepcopy(prims, g.prims, g.group_size);
    density = g.density;
    ttype = g.ttype;
    tp1x = g.tp1x;
    tp1y = g.tp1y;
    tp1z = g.tp1z;
    scale = g.scale;
    width = g.width;
    height = g.height;
    bytes_per_pixel = g.bytes_per_pixel;
    index = g.index;
    mtype = g.mtype;
    fuzz_ref_idx = g.fuzz_ref_idx;
    minx = g.minx;
    miny = g.miny;
    minz = g.minz;
    maxx = g.maxx;
    maxy = g.maxy;
    maxz = g.maxz;
    return *this;
  }
};

template <>
__host__ __device__ Vec3
min_vec<GroupParam>(const GroupParam &g) {
  Vec3 minv(g.minx, g.miny, g.minz);
  return minv;
}
template <>
__host__ __device__ Vec3
max_vec<GroupParam>(const GroupParam &g) {
  Vec3 maxv(g.maxx, g.maxy, g.maxz);
  return maxv;
}
__host__ __device__ GroupParam
makeBox(const Point3 &p0, const Point3 &p1,
        const MaterialParam &mp, int g_id) {
  HittableParam h_xyr1 = mkXYRectHittable(
      p0.x(), p1.x(), p0.y(), p1.y(), p1.z());
  Primitive side1(mp, h_xyr1, 0, g_id);

  HittableParam h_xyr2 = mkXYRectHittable(
      p0.x(), p1.x(), p0.y(), p1.y(), p0.z());
  Primitive side2(mp, h_xyr2, 1, g_id);

  HittableParam h_xzr1 = mkXZRectHittable(
      p0.x(), p1.x(), p0.z(), p1.z(), p1.y());
  Primitive side3(mp, h_xzr1, 2, g_id);

  HittableParam h_xzr2 = mkXZRectHittable(
      p0.x(), p1.x(), p0.z(), p1.z(), p0.y());
  Primitive side4(mp, h_xzr2, 3, g_id);

  HittableParam h_yzr1 = mkYZRectHittable(
      p0.y(), p1.y(), p0.z(), p1.z(), p1.x());
  Primitive side5(mp, h_yzr1, 4, g_id);

  HittableParam h_yzr2 = mkYZRectHittable(
      p0.y(), p1.y(), p0.z(), p1.z(), p0.x());
  Primitive side6(mp, h_yzr2, 5, g_id);

  Primitive ps[] = {side1, side2, side3,
                    side4, side5, side6};

  const MaterialParam mpp;
  const float g_dens = 0.0f;

  GroupParam sg(ps, 6, g_id, BOX, g_dens, mpp);
  return sg;
}

__host__ __device__ GroupParam
translate(const GroupParam &gp, Point3 steps) {
  Primitive *ps = new Primitive[gp.group_size];
  for (int i = 0; i < gp.group_size; i++) {
    Primitive p = gp.prims[i];
    ps[i] = translate(p, steps);
  }
  GroupParam g(ps, gp.group_size, gp.group_id, gp.gtype,
               gp.density, gp.get_mparam());
  delete[] ps;
  return g;
}
__host__ __device__ GroupParam rotate(const GroupParam &gp,
                                      Vec3 axis,
                                      float degree) {
  Primitive *ps = new Primitive[gp.group_size];
  for (int i = 0; i < gp.group_size; i++) {
    Primitive p = gp.prims[i];
    ps[i] = rotate(p, axis, degree);
  }
  GroupParam g(ps, gp.group_size, gp.group_id, gp.gtype,
               gp.density, gp.get_mparam());
  delete[] ps;
  return g;
}
__host__ __device__ GroupParam rotate_y(GroupParam &gp,
                                        float degree) {
  Primitive *ps = new Primitive[gp.group_size];
  for (int i = 0; i < gp.group_size; i++) {
    Primitive p = gp.prims[i];
    ps[i] = rotate_y(p, degree);
  }
  GroupParam g(ps, gp.group_size, gp.group_id, gp.gtype,
               gp.density, gp.get_mparam());
  delete[] ps;
  return g;
}
__host__ __device__ GroupParam rotate_x(GroupParam &gp,
                                        float degree) {
  Primitive *ps = new Primitive[gp.group_size];
  for (int i = 0; i < gp.group_size; i++) {
    Primitive p = gp.prims[i];
    ps[i] = rotate_x(p, degree);
  }
  GroupParam g(ps, gp.group_size, gp.group_id, gp.gtype,
               gp.density, gp.get_mparam());
  delete[] ps;
  return g;
}
__host__ __device__ GroupParam
rotate_z(const GroupParam &gp, float degree) {
  Primitive *ps = new Primitive[gp.group_size];
  for (int i = 0; i < gp.group_size; i++) {
    Primitive p = gp.prims[i];
    ps[i] = rotate_z(p, degree);
  }
  GroupParam g(ps, gp.group_size, gp.group_id, gp.gtype,
               gp.density, gp.get_mparam());
  delete[] ps;
  return g;
}
