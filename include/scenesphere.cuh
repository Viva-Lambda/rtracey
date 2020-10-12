#pragma once

#include <aabb.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <scenehit.cuh>
#include <scenematparam.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

struct Sphere {
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(Point3 cen, float r,
                             MaterialParam mat_ptr_)
      : center(cen), radius(r), mat_ptr(mat_ptr_){};
  Vec3 center;
  float radius;
  MaterialParam mat_ptr;
};

template <> struct SceneHittable<Sphere> {
  __device__ static bool hit(const Sphere &sp, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec) {
    Vec3 oc = r.origin() - sp.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - sp.radius * sp.radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - sp.center) / sp.radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = sp.mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - sp.center) / sp.radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = sp.mat_ptr;
        return true;
      }
    }
    return false;
  }
  __host__ __device__ static bool
  bounding_box(const Sphere &sp, float t0, float t1,
               Aabb &output_box) {
    output_box = Aabb(sp.center - Vec3(sp.radius),
                      sp.center + Vec3(sp.radius));
    return true;
  }
  __device__ static float pdf_value(const Sphere &sp,
                                    const Point3 &orig,
                                    const Point3 &v) {
    HitRecord rec;
    if (!SceneHittable<Sphere>::hit(sp, Ray(orig, v), 0.001,
                                    FLT_MAX, rec))
      return 0.0f;

    float rad2 = sp.radius * sp.radius;
    Vec3 cent_diff = sp.center - orig;
    auto cos_theta_max =
        sqrt(1 - rad2 / cent_diff.squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }
  __device__ static Vec3 random(const Sphere &sp,
                                const Point3 &orig,
                                curandState *loc) {
    Vec3 direction = sp.center - orig;
    auto distance_squared = direction.squared_length();
    Onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(sp.radius, distance_squared, loc));
  }
};

struct MovingSphere {
  Point3 center1, center2;
  float time0, time1, radius;
  MaterialParam mat_ptr;

  __host__ __device__ MovingSphere();
  __host__ __device__ MovingSphere(Point3 c1, Point3 c2,
                                   float t0, float t1,
                                   float rad,
                                   MaterialParam mat)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad), mat_ptr(mat) {}
  __host__ __device__ Point3 center(float time) const {
    return center1 +
           ((time - time0) / (time1 - time0)) *
               (center2 - center1);
  }
};

template <> struct SceneHittable<MovingSphere> {
  __device__ static bool hit(const MovingSphere &sp,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    Point3 scenter = sp.center(r.time());
    Vec3 oc = r.origin() - scenter;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - sp.radius * sp.radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / sp.radius;
        rec.mat_ptr = sp.mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / sp.radius;
        rec.mat_ptr = sp.mat_ptr;
        return true;
      }
    }
    return false;
  }

  __host__ __device__ static bool
  bounding_box(const MovingSphere &sp, float t0, float t1,
               Aabb &output_box) {
    output_box = Aabb(sp.center1 - Vec3(sp.radius),
                      sp.center1 + Vec3(sp.radius));
    return true;
  }
  __device__ static float pdf_value(const MovingSphere &sp,
                                    const Point3 &orig,
                                    const Point3 &v) {
    HitRecord rec;
    if (!SceneHittable<MovingSphere>::hit(
            sp, Ray(orig, v), 0.001, FLT_MAX, rec))
      return 0.0f;
    float rad2 = sp.radius * sp.radius;
    Vec3 cent_diff =
        (sp.center(sp.time1 - sp.time0) - orig);

    auto cos_theta_max =
        sqrt(1 - rad2 / cent_diff.squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }

  __device__ static Vec3 random(const MovingSphere &sp,
                                const Point3 &orig,
                                curandState *loc) {
    Vec3 direction = sp.center(sp.time1 - sp.time0) - orig;
    auto distance_squared = direction.squared_length();
    Onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(sp.radius, distance_squared, loc));
  }
};
