#pragma once

#include <aabb.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <vec3.cuh>

__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

struct Sphere : public Hittable {
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(Point3 cen, float r,
                             Material *mat_ptr_)
      : center(cen), radius(r), mat_ptr(mat_ptr_){};
  __device__ bool hit(const Ray &r, float d_min,
                      float d_max, HitRecord &rec) const {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - center) / radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - center) / radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
      }
    }
    return false;
  }
  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box =
        Aabb(center - Vec3(radius), center + Vec3(radius));
    return true;
  }
  __device__ float
  pdf_value(const Point3 &orig,
            const Point3 &v) const override {
    HitRecord rec;
    if (!hit(Ray(o, v), 0.001, FLT_MAX, rec))
      return 0.0f;

    auto cos_theta_max = sqrt(
        1 -
        radius * radius / (center - o).squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }
  __device__ Vec3 random(const Point3 &orig,
                         curandState *loc) const override {
    Vec3 direction = center - orig;
    auto distance_squared = direction.squared_length();
    Onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(radius, distance_squared, loc));
  }

  Vec3 center;
  float radius;
  MaterialParam mat_ptr;
};

template <> struct SceneHittable<Sphere> {
  __device__ static bool hit(const Sphere &sp, const Ray &r,
                             float t0, float t1,
                             HitRecord &rec) {
    return sp.hit(r, t0, t1, rec);
  }
  __host__ __device__ static bool
  bounding_box(const Sphere &sp, float t0, float t1,
               Aabb &output_box) {
    return sp.bounding_box(t0, t1, output_box);
  }
  __device__ static float pdf_value(const Sphere &sp,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return sp.pdf_value(orig, v);
  }
  __device__ static Vec3 random(const Sphere &sp,
                                const Point3 &orig,
                                curandState *loc) {
    return sp.random(orig, loc);
  }
};

struct MovingSphere : public Hittable {
  Point3 center1, center2;
  float time0, time1, radius;
  MaterialParam mat_ptr;

  __host__ __device__ MovingSphere();
  __host__ __device__ MovingSphere(Point3 c1, Point3 c2,
                                   float t0, float t1,
                                   float rad, Material *mat)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad), mat_ptr(mat) {}
  __host__ __device__ Point3 center(float time) const {
    return center1 +
           ((time - time0) / (time1 - time0)) *
               (center2 - center1);
  }
  __device__ bool hit(const Ray &r, float d_min,
                      float d_max,
                      HitRecord &rec) const override {
    Point3 scenter = center(r.time());
    Vec3 oc = r.origin() - scenter;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
      float temp = (-b - sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / radius;
        rec.mat_ptr = mat_ptr;
        return true;
      }
    }
    return false;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    output_box = Aabb(center1 - Vec3(radius),
                      center1 + Vec3(radius));
    return true;
  }
  __device__ float
  pdf_value(const Point3 &orig,
            const Point3 &v) const override {
    HitRecord rec;
    if (!hit(Ray(o, v), 0.001, FLT_MAX, rec))
      return 0.0f;

    auto cos_theta_max = sqrt(
        1 -
        radius * radius /
            (center(time1 - time0) - o).squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }
  __device__ Vec3 random(const Point3 &orig,
                         curandState *loc) const override {
    Vec3 direction = center(time1 - time0) - orig;
    auto distance_squared = direction.squared_length();
    Onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(radius, distance_squared, loc));
  }
};

template <> struct SceneHittable<MovingSphere> {
  __device__ static bool hit(const MovingSphere &sp,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return sp.hit(r, t0, t1, rec);
  }
  __host__ __device__ static bool
  bounding_box(const MovingSphere &sp, float t0, float t1,
               Aabb &output_box) {
    return sp.bounding_box(t0, t1, output_box);
  }
  __device__ static float pdf_value(const MovingSphere &sp,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return sp.pdf_value(orig, v);
  }
  __device__ static Vec3 random(const MovingSphere &sp,
                                const Point3 &orig,
                                curandState *loc) {
    return sp.random(orig, loc);
  }
};
