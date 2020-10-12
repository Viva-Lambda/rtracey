#pragma once

#include <aabb.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <vec3.cuh>

__host__ __device__ float get_pdf_surface(Vec3 dir,
                                          Vec3 normal,
                                          float dist,
                                          float area) {
  //
  float dist_squared = dist * dist / dir.squared_length();
  float cosine = fabs(dot(dir, normal) / dir.length());
  return dist_squared / (cosine * area);
}

struct AaRect : Hittable {
  Vec3 axis_normal;
  float a0, a1, // aligned1
      b0, b1;   // aligned2
  AxisInfo ax;

  float k;
  MaterialParam mat_ptr;

  __host__ __device__ AaRect() {}
  __host__ __device__ AaRect(float a_0, float a_1,
                             float b_0, float b_1, float _k,
                             MaterialParam mat,
                             Vec3 anormal)
      : a0(a_0), a1(a_1), b0(b_0), b1(b_1), k(_k),
        mat_ptr(mat), axis_normal(anormal) {
    ax = AxisInfo(axis_normal);
  }

  __device__ bool hit(const Ray &r, float t0, float t1,
                      HitRecord &rec) const override {
    /*
       point of intersection satisfies
       both P = O + D*m Point = Origin + Direction *
       magnitude
       and
       x0 < x_i < x1 y0 < y_i < y1 y0,y1 and x0,x1 being the
       limits of
       rectangle
     */
    float t = (k - r.origin()[ax.notAligned]) /
              r.direction()[ax.notAligned];
    if (t < t0 || t > t1)
      return false;
    float a = r.origin()[ax.aligned1] +
              t * r.direction()[ax.aligned1];
    float b = r.origin()[ax.aligned2] +
              t * r.direction()[ax.aligned2];
    bool c1 = rect.a0 < a and a < rect.a1;
    bool c2 = rect.b0 < b and b < rect.b1;
    if ((c1 and c2) == false) {
      return false;
    }
    rec.u = (a - a0) / (a1 - a0);
    rec.v = (b - b0) / (b1 - b0);
    rec.t = t;
    Vec3 outward_normal = axis_normal;
    rec.set_front_face(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    rec.p = r.at(t);
    return true;
  }

  __host__ __device__ bool
  bounding_box(float t0, float t1,
               Aabb &output_box) const override {
    // The bounding box must have non-zero width in each
    // dimension, so pad the Z
    // dimension a small amount.
    Point3 p1, p2;
    // choose points with axis
    switch (ax.notAligned) {
    case 2: {
      p1 = Point3(a0, b0, k - 0.0001);
      p2 = Point3(a1, b1, k + 0.0001);
      break;
    }
    case 1: {
      p1 = Point3(a0, k - 0.0001, b0);
      p2 = Point3(a1, k + 0.0001, b1);
      break;
    }
    case 0: {
      p1 = Point3(k - 0.0001, a0, b0);
      p2 = Point3(k + 0.0001, a1, b1);
      break;
    }
    }
    output_box = Aabb(p1, p2);
    return true;
  }
  __device__ float
  pdf_value(const Point3 &orig,
            const Point3 &v) const override {
    HitRecord rec;
    if (!hit(Ray(orig, v), 0.001, FLT_MAX, rec))
      return 0;

    float area = (rect.a1 - rect.a0) * (rect.b1 - rect.b0);
    return get_pdf_surface(v, rec.normal, rec.t, area);
  }
  __device__ Vec3 random(const Point3 &orig,
                         curandState *loc) const override {
    Point3 random_point =
        Point3(random_float(loc, a0, a1), k,
               random_float(loc, b0, b1));
    return random_point - orig;
  }
};

template <> struct SceneHittable<AaRect> {
  __device__ static bool hit(const AaRect &rect,
                             const Ray &r, float t0,
                             float t1, HitRecord &rec) {
    return rect.hit(r, t0, t1, rec);
  }
  __host__ __device__ static bool
  bounding_box(const AaRect &rect, float t0, float t1,
               Aabb &output_box) {
    return rect.bounding_box(t0, t1, output_box);
  }
  __device__ static float pdf_value(const AaRect &rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    return rect.pdf_value(orig, v);
  }
  __device__ static Vec3 random(const AaRect &rect,
                                const Point3 &orig,
                                curandState *loc) {
    return rect.random(orig, loc);
  }
};

struct XYRect : public AaRect {
  float x0, x1, y0, y1;

  __host__ __device__ XYRect() {}
  __host__ __device__ XYRect(float _x0, float _x1,
                             float _y0, float _y1, float _k,
                             MaterialParam mat)
      : AaRect(_x0, _x1, _y0, _y1, _k, mat, Vec3(0, 0, 1)),
        x0(_x0), x1(_x1), y0(_y0), y1(_y1) {}
};
struct XZRect : public AaRect {
  float x0, x1, z0, z1;

  __host__ __device__ XZRect() {}
  __host__ __device__ XZRect(float _x0, float _x1,
                             float _z0, float _z1, float _k,
                             MaterialParam mat)
      : AaRect(_x0, _x1, _z0, _z1, _k, mat, Vec3(0, 1, 0)),
        x0(_x0), x1(_x1), z0(_z0), z1(_z1) {}
};
struct YZRect : public AaRect {
  float y0, y1, z0, z1;

  __host__ __device__ YZRect() {}
  __host__ __device__ YZRect(float _y0, float _y1,
                             float _z0, float _z1, float _k,
                             MaterialParam mat)
      : AaRect(_y0, _y1, _z0, _z1, _k, mat, Vec3(1, 0, 0)),
        y0(_y0), y1(_y1), z0(_z0), z1(_z1) {}
};
