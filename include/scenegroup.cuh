#ifndef SCENEGROUP_CUH
#define SCENEGROUP_CUH

#include <mediumc.cuh>
#include <ray.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

struct SceneGroup {
  // group params
  GroupType gtype;
  int group_size;
  int group_id;
  ScenePrim *prims;
  //
  float density;
  TextureParam tparam;

  __host__ __device__ SceneGroup() {}
  __host__ __device__ SceneGroup(ScenePrim *prm, int gsize,
                                 int gid, GroupType gtp,
                                 float d, TextureParam tp)
      : prims(prm), group_size(gsize), group_id(gid),
        gtype(gtp), density(d), tparam(tp) {}
  __host__ __device__ inline bool get(int i,
                                      ScenePrim &p) const {
    if (i > 0 && i < group_size) {
      p = prims[i];
      return true;
    }
    return false;
  }
};

template <> struct SceneHittable<SceneGroup> {
  __device__ static bool hit(const SceneGroup &g,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < g.group_size; i++) {
      ScenePrim p;
      g.get(i, p);
      bool isHit = SceneHittable<ScenePrim>::hit(
          p, r, d_min, closest_far, temp);
      if (isHit == true) {
        hit_anything = isHit;
        closest_far = temp.t;
        rec = temp;
        rec.group_id = g.group_id;
        rec.group_index = p.group_index;
      }
    }
    return hit_anything;
  }
  __host__ __device__ static bool
  bounding_box(SceneGroup g, float t0, float t1,
               Aabb &output_box) {
    Aabb temp;
    bool first_box = true;
    for (int i = 0; i < g.group_size; i++) {
      ScenePrim p;
      g.get(i, p);

      bool isBounding =
          SceneHittable<ScenePrim>::bounding_box(p, t0, t1,
                                                 temp);
      if (isBounding == false) {
        return false;
      }
      output_box = first_box
                       ? temp
                       : surrounding_box(output_box, temp);
      first_box = false;
      // center = output_box.center;
    }
    return true;
  }
  __device__ static float pdf_value(SceneGroup g,
                                    const Point3 &o,
                                    const Point3 &v) {
    //
    float weight = 1.0f / g.group_size;
    float sum = 0.0f;
    for (int i = 0; i < g.group_size; i++) {
      ScenePrim p;
      g.get(i, p);
      sum += weight *
             SceneHittable<ScenePrim>::pdf_value(p, o, v);
    }
    return sum;
  }

  __device__ static Vec3 random(SceneGroup g, const Vec3 &v,
                                curandState *loc) {
    int obj_index = random_int(loc, 0, g.group_size - 1);
    ScenePrim p;
    g.get(obj_index, p);
    return SceneHittable<ScenePrim>::random(p, v, loc);
  }
};

#endif
