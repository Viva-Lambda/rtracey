#pragma once
#include <aabb.cuh>
#include <external.hpp>
#include <onb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <shape.cuh>
#include <utils.cuh>
#include <vec3.cuh>
#include <scenetype.cuh>
__host__ __device__ void get_sphere_uv(const Vec3 &p,
                                       float &u, float &v) {
  auto phi = atan2(p.z(), p.x());
  auto theta = asin(p.y());
  u = 1 - (phi + M_PI) / (2 * M_PI);
  v = (theta + M_PI / 2) / M_PI;
}

template <class HiT> struct SceneHittable {

  __device__ static bool hit(const HiT &h, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec);
  __host__ __device__ static bool
  bounding_box(const HiT &h, float t0, float t1,
               Aabb &output_box);
  __device__ static float pdf_value(const HiT &&h,
                                    const Point3 &o,
                                    const Point3 &v);
  __device__ static Vec3 random(const HiT &h, const Vec3 &v,
                                curandState *loc) {
    return Vec3(1.0f, 0.0f, 0.0f);
  }
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
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        Vec3 normal = (rec.p - sp.center) / sp.radius;
        rec.set_front_face(r, normal);
        get_sphere_uv(normal, rec.u, rec.v);
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
template <> struct SceneHittable<MovingSphere> {
  __device__ static bool hit(const MovingSphere &sp,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    float rt = r.time();
    Point3 scenter = sp.center(rt);
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
        return true;
      }
      temp = (-b + sqrt(discriminant)) / a;
      if (temp < d_max && temp > d_min) {
        rec.t = temp;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - scenter) / sp.radius;
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
    float tdiff = sp.time1 - sp.time0;
    Vec3 cent_diff = sp.center(tdiff) - orig;

    auto cos_theta_max =
        sqrt(1 - rad2 / cent_diff.squared_length());
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1 / solid_angle;
  }

  __device__ static Vec3 random(const MovingSphere &sp,
                                const Point3 &orig,
                                curandState *loc) {

    float tdiff = sp.time1 - sp.time0;
    Vec3 direction = sp.center(tdiff) - orig;
    auto distance_squared = direction.squared_length();
    Onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(
        random_to_sphere(sp.radius, distance_squared, loc));
  }
};
template <> struct SceneHittable<Triangle> {
  __device__ static bool hit(const Triangle &tri,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    // implementing moller from wikipedia
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    const float eps = 0.000001f;
    Vec3 edge1 = tri.p1 - tri.p2;
    Vec3 edge2 = tri.p3 - tri.p2;
    Vec3 h = cross(r.direction(), edge2);
    float a = dot(edge1, h);
    if (a > eps && a < eps)
      return false; // ray parallel to triangle
    float f = 1.0f / a;
    Vec3 rToP2 = r.origin() - tri.p2;
    float u = f * dot(rToP2, h);
    if (u < 0.0f || u > 1.0f)
      return false;

    Vec3 q = cross(rToP2, edge1);
    float v = f * dot(edge2, q);
    if (v < 0.0f || v > 1.0f)
      return false;

    float t = f * dot(r.direction(), q);
    if (t < eps)
      return false;

    rec.v = v;
    rec.u = u;
    rec.t = t;
    rec.p = r.at(rec.t);
    Vec3 outnormal = cross(edge1, edge2);
    rec.set_front_face(r, outnormal);
    return true;
  }
  __host__ __device__ static bool
  bounding_box(const Triangle &tri, float t0, float t1,
               Aabb &output_box) {
    Point3 pmin = min_vec(tri.p1, tri.p2);
    pmin = min_vec(pmin, tri.p3);

    Point3 pmax = max_vec(tri.p1, tri.p2);
    pmax = max_vec(pmax, tri.p3);
    output_box = Aabb(pmin, pmax);
    return true;
  }
  __device__ static float pdf_value(const Triangle &tri,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return 1.0f;
  }

  __device__ static Vec3 random(const Triangle &tri,
                                const Point3 &o,
                                curandState *loc) {
    // from A. Glassner, Graphics Gems, 1995, p. 24
    float t = curand_uniform(loc);
    float s = curand_uniform(loc);
    auto a = 1 - sqrt(t);
    auto b = (1 - s) * sqrt(t);
    auto c = s * sqrt(t);
    auto random_point =
        a * tri.p1 + b * tri.p2 + c * tri.p3;
    return random_point - o;
  }
};
template <> struct SceneHittable<AaTriangle> {
  __device__ static bool hit(const AaTriangle &tri,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<Triangle>::hit(tri, r, t0, t1,
                                        rec);
  }
  __host__ __device__ static bool
  bounding_box(const AaTriangle &tri, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<Triangle>::bounding_box(
        tri, t0, t1, output_box);
  }
  __device__ static Vec3 random(const AaTriangle &tri,
                                const Point3 &o,
                                curandState *loc) {
    return SceneHittable<Triangle>::random(tri, o, loc);
  }
};
template <> struct SceneHittable<XYTriangle> {
  __device__ static bool hit(const XYTriangle &tri,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaTriangle>::hit(tri, r, t0, t1,
                                          rec);
  }
  __host__ __device__ static bool
  bounding_box(const XYTriangle &tri, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaTriangle>::bounding_box(
        tri, t0, t1, output_box);
  }
  __device__ static Vec3 random(const XYTriangle &tri,
                                const Point3 &o,
                                curandState *loc) {
    return SceneHittable<AaTriangle>::random(tri, o, loc);
  }
};
template <> struct SceneHittable<XZTriangle> {
  __device__ static bool hit(const XZTriangle &tri,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaTriangle>::hit(tri, r, t0, t1,
                                          rec);
  }
  __host__ __device__ static bool
  bounding_box(const XZTriangle &tri, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaTriangle>::bounding_box(
        tri, t0, t1, output_box);
  }
  __device__ static Vec3 random(const XZTriangle &tri,
                                const Point3 &o,
                                curandState *loc) {
    return SceneHittable<AaTriangle>::random(tri, o, loc);
  }
};
template <> struct SceneHittable<YZTriangle> {
  __device__ static bool hit(const YZTriangle &tri,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaTriangle>::hit(tri, r, t0, t1,
                                          rec);
  }
  __host__ __device__ static bool
  bounding_box(const YZTriangle &tri, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaTriangle>::bounding_box(
        tri, t0, t1, output_box);
  }
  __device__ static Vec3 random(const YZTriangle &tri,
                                const Point3 &o,
                                curandState *loc) {
    return SceneHittable<AaTriangle>::random(tri, o, loc);
  }
};
__host__ __device__ float get_pdf_surface(Vec3 dir,
                                          Vec3 normal,
                                          float dist,
                                          float area) {
  float dist_squared = dist * dist / dir.squared_length();
  float cosine = fabs(dot(dir, normal) / dir.length());
  return dist_squared / (cosine * area);
}
template <> struct SceneHittable<AaRect> {

  __device__ static bool hit(const AaRect &rect,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    /*
       point of intersection satisfies
       both P = O + D*m Point = Origin + Direction *
       magnitude
       and
       x0 < x_i < x1 y0 < y_i < y1 y0,y1 and x0,x1 being the
       limits of
       rectangle
     */
    float t = (rect.k - r.origin()[rect.ax.notAligned]) /
              r.direction()[rect.ax.notAligned];
    if (t < t0 || t > t1)
      return false;
    float a = r.origin()[rect.ax.aligned1] +
              t * r.direction()[rect.ax.aligned1];
    float b = r.origin()[rect.ax.aligned2] +
              t * r.direction()[rect.ax.aligned2];
    bool c1 = rect.a0 < a and a < rect.a1;
    bool c2 = rect.b0 < b and b < rect.b1;
    if ((c1 and c2) == false) {
      return false;
    }
    rec.u = (a - rect.a0) / (rect.a1 - rect.a0);
    rec.v = (b - rect.b0) / (rect.b1 - rect.b0);
    rec.t = t;
    Vec3 outward_normal = rect.axis_normal;
    rec.set_front_face(r, outward_normal);
    rec.p = r.at(t);
    return true;
  }

  __host__ __device__ static bool
  bounding_box(const AaRect &rect, float t0, float t1,
               Aabb &output_box) {
    // The bounding box must have non-zero width in each
    // dimension, so pad the Z
    // dimension a small amount.
    Point3 p1, p2;
    // choose points with axis
    switch (rect.ax.notAligned) {
    case 2: {
      p1 = Point3(rect.a0, rect.b0, rect.k - 0.0001);
      p2 = Point3(rect.a1, rect.b1, rect.k + 0.0001);
      break;
    }
    case 1: {
      p1 = Point3(rect.a0, rect.k - 0.0001, rect.b0);
      p2 = Point3(rect.a1, rect.k + 0.0001, rect.b1);
      break;
    }
    case 0: {
      p1 = Point3(rect.k - 0.0001, rect.a0, rect.b0);
      p2 = Point3(rect.k + 0.0001, rect.a1, rect.b1);
      break;
    }
    }
    output_box = Aabb(p1, p2);
    return true;
  }

  __device__ static float pdf_value(const AaRect &rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    HitRecord rec;
    if (!SceneHittable<AaRect>::hit(rect, Ray(orig, v),
                                    0.001, FLT_MAX, rec))
      return 0;

    float area = (rect.a1 - rect.a0) * (rect.b1 - rect.b0);
    return get_pdf_surface(v, rec.normal, rec.t, area);
  }

  __device__ static Vec3 random(const AaRect &rect,
                                const Point3 &orig,
                                curandState *loc) {
    Point3 random_point =
        Point3(random_float(loc, rect.a0, rect.a1), rect.k,
               random_float(loc, rect.b0, rect.b1));
    return random_point - orig;
  }
};
template <> struct SceneHittable<XYRect> {

  __device__ static bool hit(const XYRect &rect,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaRect>::hit(rect, r, t0, t1, rec);
  }

  __host__ __device__ static bool
  bounding_box(const XYRect &rect, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaRect>::bounding_box(rect, t0, t1,
                                               output_box);
  }

  __device__ static float pdf_value(const XYRect &rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return SceneHittable<AaRect>::pdf_value(rect, orig, v);
  }
  __device__ static Vec3 random(const XYRect &rect,
                                const Point3 &orig,
                                curandState *loc) {
    return SceneHittable<AaRect>::random(rect, orig, loc);
  }
};
template <> struct SceneHittable<XZRect> {
  __device__ static bool hit(const XZRect &rect,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaRect>::hit(rect, r, t0, t1, rec);
  }
  __host__ __device__ static bool
  bounding_box(const XZRect &rect, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaRect>::bounding_box(rect, t0, t1,
                                               output_box);
  }
  __device__ static float pdf_value(const XZRect &rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return SceneHittable<AaRect>::pdf_value(rect, orig, v);
  }
  __device__ static Vec3 random(const XZRect &rect,
                                const Point3 &orig,
                                curandState *loc) {
    return SceneHittable<AaRect>::random(rect, orig, loc);
  }
};
template <> struct SceneHittable<YZRect> {
  __device__ static bool hit(const YZRect &rect,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return SceneHittable<AaRect>::hit(rect, r, t0, t1, rec);
  }
  __host__ __device__ static bool
  bounding_box(const YZRect &rect, float t0, float t1,
               Aabb &output_box) {
    return SceneHittable<AaRect>::bounding_box(rect, t0, t1,
                                               output_box);
  }
  __device__ static float pdf_value(const YZRect &rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return SceneHittable<AaRect>::pdf_value(rect, orig, v);
  }
  __device__ static Vec3 random(const YZRect &rect,
                                const Point3 &orig,
                                curandState *loc) {
    return SceneHittable<AaRect>::random(rect, orig, loc);
  }
};
