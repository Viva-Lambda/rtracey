#pragma once
#include <aabb.cuh>
#include <group.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <sceneprim.cuh>
#include <sceneshape.cuh>
#include <vec3.cuh>

template <> struct SceneHittable<ConstantMedium> {
  __device__ static bool hit(const ConstantMedium &g,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    // Print occasional samples when debugging. To enable,
    // set enableDebug true.
    const bool enableDebug = false;

    const bool debugging =
        enableDebug && curand_uniform(g.rState) < 0.00001;

    HitRecord rec1, rec2;

    bool any_hit = false;
    for (int i = 0; i < g.nb_prims; i++) {
      Primitive p = g.boundary[i];
      any_hit = SceneHittable<Primitive>::hit(
          p, r, -FLT_MAX, FLT_MAX, rec1);
    }
    if (!any_hit)
      return any_hit;

    for (int i = 0; i < g.nb_prims; i++) {
      Primitive p = g.boundary[i];
      any_hit = SceneHittable<Primitive>::hit(
          p, r, rec1.t + 0.0001, FLT_MAX, rec2);
    }
    if (!any_hit)
      return any_hit;

    if (debugging) {
      printf("\nt0= %f", rec1.t);
      printf(", t1= %f\n", rec2.t);
    }

    if (rec1.t < d_min)
      rec1.t = d_min;
    if (rec2.t > d_max)
      rec2.t = d_max;

    if (rec1.t >= rec2.t)
      return false;

    if (rec1.t < 0)
      rec1.t = 0;

    const float ray_length = r.direction().length();
    const float distance_inside_boundary =
        (rec2.t - rec1.t) * ray_length;
    const float hit_distance =
        g.neg_inv_density * log(curand_uniform(g.rState));

    if (hit_distance > distance_inside_boundary)
      return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    if (debugging) {
      printf("hit_distance = %f\n", hit_distance);
      printf("rec.t = %f\n", rec.t);
      printf("rec.p = %f ", rec.p.x());
      printf("%f ", rec.p.y());
      printf("%f ", rec.p.z());
    }

    rec.normal = Vec3(1, 0, 0); // arbitrary
    rec.front_face = true;      // also arbitrary

    return true;
  }
  __host__ __device__ static bool
  bounding_box(const ConstantMedium &g, float t0, float t1,
               Aabb &output_box) {
    Aabb temp;
    bool first_box = true;
    for (int i = 0; i < g.nb_prims; i++) {
      Primitive p = g.boundary[i];
      bool isBounding =
          SceneHittable<Primitive>::bounding_box(p, t0, t1,
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
  __device__ static float pdf_value(const ConstantMedium &g,
                                    const Point3 &o,
                                    const Point3 &v) {
    float weight = 1.0f / g.nb_prims;
    float sum = 0.0f;
    for (int i = 0; i < g.nb_prims; i++) {
      Primitive p = g.boundary[i];
      sum += weight *
             SceneHittable<Primitive>::pdf_value(p, o, v);
    }
    return sum;
  }
  __device__ static Vec3 random(const ConstantMedium &g,
                                const Vec3 &v,
                                curandState *loc) {
    int obj_index = random_int(loc, 0, g.nb_prims - 1);
    Primitive p = g.boundary[obj_index];
    return SceneHittable<Primitive>::random(p, v, loc);
  }
};
template <> struct SceneHittable<Box> {
  __device__ static bool hit(const Box &g, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec) {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < g.group_size; i++) {
      Primitive p = g.prims[i];
      bool isHit = SceneHittable<Primitive>::hit(
          p, r, d_min, closest_far, temp);
      if (isHit == true) {
        hit_anything = isHit;
        closest_far = temp.t;
        rec = temp;
      }
    }
    return hit_anything;
  }

  __host__ __device__ static bool
  bounding_box(const Box &g, float t0, float t1,
               Aabb &output_box) {
    Point3 min_p;
    Point3 max_p;
    g.minmax_points(min_p, max_p);
    output_box = Aabb(min_p, max_p);
    return true;
  }

  __device__ static float pdf_value(const Box &g,
                                    const Point3 &o,
                                    const Point3 &v) {
    float weight = 1.0f / g.group_size;
    float sum = 0.0f;
    for (int i = 0; i < g.group_size; i++) {
      Primitive p = g.prims[i];
      sum += weight *
             SceneHittable<Primitive>::pdf_value(p, o, v);
    }
    return sum;
  }

  __device__ static Vec3 random(const Box &g, const Vec3 &v,
                                curandState *loc) {
    int obj_index = random_int(loc, 0, g.group_size - 1);
    Primitive p = g.prims[obj_index];
    return SceneHittable<Primitive>::random(p, v, loc);
  }
};
template <> struct SceneHittable<SimpleMesh> {
  __device__ static bool hit(const SimpleMesh &g,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    HitRecord temp;
    bool hit_anything = false;
    float closest_far = d_max;
    for (int i = 0; i < g.group_size; i++) {
      Primitive p = g.prims[i];
      bool isHit = SceneHittable<Primitive>::hit(
          p, r, d_min, closest_far, temp);
      if (isHit == true) {
        hit_anything = isHit;
        closest_far = temp.t;
        rec = temp;
        rec.group_index = p.group_index;
      }
    }
    return hit_anything;
  }
  __host__ __device__ static bool
  bounding_box(const SimpleMesh &g, float t0, float t1,
               Aabb &output_box) {
    Aabb temp;
    bool first_box = true;
    for (int i = 0; i < g.group_size; i++) {
      Primitive p = g.prims[i];
      bool isBounding =
          SceneHittable<Primitive>::bounding_box(p, t0, t1,
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
  __device__ static float pdf_value(const SimpleMesh &g,
                                    const Point3 &o,
                                    const Point3 &v) {
    float weight = 1.0f / g.group_size;
    float sum = 0.0f;
    for (int i = 0; i < g.group_size; i++) {
      Primitive p = g.prims[i];
      sum += weight *
             SceneHittable<Primitive>::pdf_value(p, o, v);
    }
    return sum;
  }

  __device__ static Vec3 random(const SimpleMesh &g,
                                const Vec3 &v,
                                curandState *loc) {
    int obj_index = random_int(loc, 0, g.group_size - 1);
    Primitive p = g.prims[obj_index];
    return SceneHittable<Primitive>::random(p, v, loc);
  }
};
