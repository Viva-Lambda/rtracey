#pragma once
#include <group.cuh>
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

  __host__ __device__ GroupParam()
      : gtype(NONE_GRP), group_size(0), group_id(0),
        density(0.0f), prims(nullptr), width(0), height(0),
        bytes_per_pixel(0), index(0), tp1x(0), tp1y(0),
        tp1z(0), scale(0), ttype(NONE_TEXTURE) {}
  __host__ __device__ GroupParam(Primitive *prm,
                                 const int gsize,
                                 const int gid,
                                 const GroupType gtp,
                                 const float d,
                                 const TextureParam &tp)
      : group_size(gsize), group_id(gid), gtype(gtp),
        density(d), width(tp.width), height(tp.height),
        bytes_per_pixel(tp.bytes_per_pixel),
        index(tp.index), tp1x(tp.tp1x), tp1y(tp.tp1y),
        tp1z(tp.tp1z), scale(tp.scale), ttype(tp.ttype) {
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
    //
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
    return *this;
  }
};

__host__ __device__ GroupParam makeBox(const Point3 &p0,
                                       const Point3 &p1,
                                       MaterialParam mp,
                                       int g_id) {
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

  const TextureParam tp;
  const float g_dens = 0.0f;

  GroupParam sg(ps, 6, g_id, BOX, g_dens, tp);
  return sg;
}
