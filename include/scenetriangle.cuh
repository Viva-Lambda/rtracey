#pragma once
#include <aabb.cuh>
#include <ray.cuh>
#include <record.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <utils.cuh>
#include <vec3.cuh>

struct Triangle {
  Point3 p1, p2, p3;
  MaterialParam mat_ptr;
  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(Point3 p_1, Point3 p_2,
                               Point3 p_3,
                               MaterialParam mpram)
      : p1(p_1), p2(p_2), p3(p_3), mat_ptr(mpram) {}
};

struct AaTriangle : Triangle {
  Vec3 axis_normal;
  AxisInfo ax;
  MaterialParam mat_ptr;
  Point3 p1, p2, p3;
  __host__ __device__ AaTriangle() {}
  __host__ __device__ AaTriangle(float a0, float a1,
                                 float a2, float b0,
                                 float b1, float k,
                                 MaterialParam mp,
                                 Vec3 anormal)
      : mat_ptr(mp), ax(AxisInfo(anormal)),
        axis_normal(anormal) {
    switch (ax.notAligned) {
    case 2: {
      p1 = Point3(a0, b0, k);
      p2 = Point3(a1, b0, k);
      p3 = Point3(a2, b1, k);
      break;
    }
    case 1: {
      p1 = Point3(a0, k, b0);
      p2 = Point3(a1, k, b0);
      p3 = Point3(a2, k, b1);
      break;
    }
    case 0: {
      p1 = Point3(k, a0, b0);
      p2 = Point3(k, a1, b0);
      p3 = Point3(k, a2, b1);
      break;
    }
    }
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
    rec.mat_ptr = tri.mat_ptr;
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

struct XYTriangle : AaTriangle {

  __host__ __device__ XYTriangle() {}
  __host__ __device__ XYTriangle(float x0, float x1,
                                 float x2, float y0,
                                 float y1, float z,
                                 MaterialParam mp)
      : AaTriangle(x0, x1, x2, y0, y1, z, mp,
                   Vec3(0, 0, 1)) {}
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
struct XZTriangle : AaTriangle {

  __host__ __device__ XZTriangle() {}
  __host__ __device__ XZTriangle(float x0, float x1,
                                 float x2, float z0,
                                 float z1, float y,
                                 MaterialParam mp)
      : AaTriangle(x0, x1, x2, z0, z1, y, mp,
                   Vec3(0, 1, 0)) {}
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
struct YZTriangle : AaTriangle {

  __host__ __device__ YZTriangle() {}
  __host__ __device__ YZTriangle(float y0, float y1,
                                 float y2, float z0,
                                 float z1, float x,
                                 MaterialParam mp)
      : AaTriangle(y0, y1, y2, z0, z1, x, mp,
                   Vec3(1, 0, 0)) {}
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
