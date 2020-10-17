#pragma once

#include <groupparam.cuh>
#include <ray.cuh>
#include <scenegroup.cuh>
#include <sceneprim.cuh>

template <> struct SceneHittable<GroupParam> {
  __device__ static bool hit(const GroupParam &g,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    bool res = false;
    const int gtype = g.gtype;
    if (gtype == BOX) {
      Box bg = g.to_box();
      res =
          SceneHittable<Box>::hit(bg, r, d_min, d_max, rec);
    } else if (gtype == CONSTANT_MEDIUM) {
      ConstantMedium bg = g.to_constant_medium();
      res = SceneHittable<ConstantMedium>::hit(bg, r, d_min,
                                               d_max, rec);
    } else if (gtype == SIMPLE_MESH) {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::hit(bg, r, d_min,
                                           d_max, rec);
    }
    return res;
  }
  __host__ __device__ static bool
  bounding_box(const GroupParam &g, float t0, float t1,
               Aabb &output_box) {
    bool res = false;
    const int gtype = g.gtype;
    if (gtype == BOX) {
      Box bg = g.to_box();
      res = SceneHittable<Box>::bounding_box(bg, t0, t1,
                                             output_box);
    } else if (gtype == CONSTANT_MEDIUM) {
      ConstantMedium bg = g.to_constant_medium();

      res = SceneHittable<ConstantMedium>::bounding_box(
          bg, t0, t1, output_box);
    } else if (gtype == SIMPLE_MESH) {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::bounding_box(
          bg, t0, t1, output_box);
    }
    return res;
  }
  __device__ static float pdf_value(const GroupParam &g,
                                    const Point3 &o,
                                    const Point3 &v) {
    float res = 0.0f;
    const int gtype = g.gtype;
    if (gtype == BOX) {
      Box bg = g.to_box();
      res = SceneHittable<Box>::pdf_value(bg, o, v);
    } else if (gtype == CONSTANT_MEDIUM) {
      ConstantMedium bg = g.to_constant_medium();
      res = SceneHittable<ConstantMedium>::pdf_value(bg, o,
                                                     v);
    } else if (gtype == SIMPLE_MESH) {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::pdf_value(bg, o, v);
    }
    return res;
  }
  __device__ static Vec3 random(const GroupParam &g,
                                const Vec3 &v,
                                curandState *loc) {
    Vec3 res(0.0f, 0.0f, 0.0f);
    const int gtype = g.gtype;
    if (gtype == BOX) {
      Box bg = g.to_box();
      res = SceneHittable<Box>::random(bg, v, loc);
    } else if (gtype == CONSTANT_MEDIUM) {
      ConstantMedium bg = g.to_constant_medium();
      res =
          SceneHittable<ConstantMedium>::random(bg, v, loc);
    } else if (gtype == SIMPLE_MESH) {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::random(bg, v, loc);
    }
    return res;
  }
};
