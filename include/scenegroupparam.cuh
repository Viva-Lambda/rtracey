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
    switch (g.gtype) {
    case BOX: {
      Box bg = g.to_box();
      res =
          SceneHittable<Box>::hit(bg, r, d_min, d_max, rec);
      break;
    }
    case CONSTANT_MEDIUM: {
      ConstantMedium bg = g.to_constant_medium();
      res = SceneHittable<ConstantMedium>::hit(bg, r, d_min,
                                               d_max, rec);
      break;
    }
    case SIMPLE_MESH: {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::hit(bg, r, d_min,
                                           d_max, rec);
      break;
    }
    }
    return res;
  }
  __host__ __device__ static bool
  bounding_box(const GroupParam &g, float t0, float t1,
               Aabb &output_box) {
    bool res = false;
    switch (g.gtype) {
    case BOX: {
      Box bg = g.to_box();
      res = SceneHittable<Box>::bounding_box(bg, t0, t1,
                                             output_box);
      break;
    }
    case CONSTANT_MEDIUM: {
      ConstantMedium bg = g.to_constant_medium();

      res = SceneHittable<ConstantMedium>::bounding_box(
          bg, t0, t1, output_box);
      break;
    }
    case SIMPLE_MESH: {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::bounding_box(
          bg, t0, t1, output_box);
      break;
    }
    }
    return res;
  }
  __device__ static float pdf_value(const GroupParam &g,
                                    const Point3 &o,
                                    const Point3 &v) {
    float res = 0.0f;
    switch (g.gtype) {
    case BOX: {
      Box bg = g.to_box();
      res = SceneHittable<Box>::pdf_value(bg, o, v);
      break;
    }
    case CONSTANT_MEDIUM: {
      ConstantMedium bg = g.to_constant_medium();
      res = SceneHittable<ConstantMedium>::pdf_value(bg, o,
                                                     v);
      break;
    }
    case SIMPLE_MESH: {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::pdf_value(bg, o, v);
      break;
    }
    }
    return res;
  }
  __device__ static Vec3 random(const GroupParam &g,
                                const Vec3 &v,
                                curandState *loc) {
    Vec3 res(0.0f);
    switch (g.gtype) {
    case BOX: {
      Box bg = g.to_box();
      res = SceneHittable<Box>::random(bg, v, loc);
      break;
    }
    case CONSTANT_MEDIUM: {
      ConstantMedium bg = g.to_constant_medium();
      res =
          SceneHittable<ConstantMedium>::random(bg, v, loc);
      break;
    }
    case SIMPLE_MESH: {
      SimpleMesh bg = g.to_simple_mesh();
      res = SceneHittable<SimpleMesh>::random(bg, v, loc);
      break;
    }
    }
    return res;
  }
};
