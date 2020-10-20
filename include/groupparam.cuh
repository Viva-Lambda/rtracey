#pragma once
#include <group.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct GroupParam {
  // group params
  const int gtype;
  const int group_size;
  const int group_id;
  Primitive *prims;
  //
  const float density;

  const TextureType ttype;
  const float tp1x, tp1y, tp1z;
  const float scale;
  const int width, height, bytes_per_pixel, index;

  __host__ __device__ GroupParam()
      : gtype(0), group_size(0), group_id(0), density(0.0f),
        prims(nullptr), width(0), height(0),
        bytes_per_pixel(0), index(0), tp1x(0), tp1y(0),
        tp1z(0), scale(0), ttype(NONE_TEXTURE) {}
  __host__ __device__ GroupParam(
      Primitive *prm, const int gsize, const int gid,
      const int gtp, const float d, const TextureParam &tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), width(tp.width),
        height(tp.height),
        bytes_per_pixel(tp.bytes_per_pixel),
        index(tp.index), tp1x(tp.tp1x), tp1y(tp.tp1y),
        tp1z(tp.tp1z), scale(tp.scale), ttype(tp.ttype) {}
  __host__ __device__ Primitive get(int i) const {
    if (i <= 0) {
      return prims[0];
    } else if (i < group_size) {
      return prims[i];
    } else {
      return prims[group_size - 1];
    }
  }
};

GroupParam makeBox(const Point3 &p0, const Point3 &p1,
                   MaterialParam mp, int g_id) {
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
  const TextureParam tp;
  const float g_dens = 0.0f;

  Primitive ps[] = {side1, side2, side3,
                    side4, side5, side6};
  GroupParam sg(ps, 6, g_id, BOX, g_dens, tp);
  return sg;
}
