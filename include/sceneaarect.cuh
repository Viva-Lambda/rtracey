#pragma once

#include <aabb.cuh>
#include <external.hpp>
#include <ray.cuh>
#include <record.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <vec3.cuh>

struct AxisInfo {
  int aligned1;
  int aligned2;
  int notAligned;
  __host__ __device__ AxisInfo() {}
  __host__ __device__ AxisInfo(Vec3 anormal) {
    if (anormal.z() == 1.0) {
      aligned1 = 0;
      aligned2 = 1;
      notAligned = 2;
    } else if (anormal.x() == 1.0) {
      aligned1 = 1;
      aligned2 = 2;
      notAligned = 0;
    } else if (anormal.y() == 1.0) {
      aligned1 = 0;
      aligned2 = 2;
      notAligned = 1;
    }
  }
};

__host__ __device__ float get_pdf_surface(Vec3 dir,
                                          Vec3 normal,
                                          float dist,
                                          float area) {
  //
  float dist_squared = dist * dist / dir.squared_length();
  float cosine = fabs(dot(dir, normal) / dir.length());
  return dist_squared / (cosine * area);
}

class AaRect {
public:
  Vec3 axis_normal;
  float a0, a1, // aligned1
      b0, b1;   // aligned2
  AxisInfo ax;

public:
  float k;
  MaterialParam mat_ptr;

public:
  __host__ __device__ AaRect() {}
  __host__ __device__ AaRect(float a_0, float a_1,
                             float b_0, float b_1, float _k,
                             MaterialParam mat,
                             Vec3 anormal)
      : a0(a_0), a1(a_1), b0(b_0), b1(b_1), k(_k),
        mat_ptr(mat), axis_normal(anormal) {
    ax = AxisInfo(axis_normal);
  }
};

template <> struct SceneHittable<AaRect> {
  //
  __device__ static bool hit(AaRect rect, const Ray &r,
                             float t0, float t1,
                             HitRecord &rec) {
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
    rec.mat_ptr = rect.mat_ptr;
    rec.p = r.at(t);
    return true;
  }
  __host__ __device__ static bool
  bounding_box(AaRect rect, float t0, float t1,
               Aabb &output_box) override {
    // The bounding box must have non-zero width in each
    // dimension, so pad the Z
    // dimension a small amount.
    Point3 p1, p2;
    // choose points with axis
    switch (rect.ax.notAligned) {
    case 2:
      p1 = Point3(rect.a0, rect.b0, rect.k - 0.0001);
      p2 = Point3(rect.a1, rect.b1, rect.k + 0.0001);
      break;
    case 1:
      p1 = Point3(rect.a0, rect.k - 0.0001, rect.b0);
      p2 = Point3(rect.a1, rect.k + 0.0001, rect.b1);
      break;
    case 0:
      p1 = Point3(rect.k - 0.0001, rect.a0, rect.b0);
      p2 = Point3(rect.k + 0.0001, rect.a1, rect.b1);
      break;
    }
    output_box = Aabb(p1, p2);
    return true;
  }
  __device__ static float pdf_value(AaRect rect,
                                    const Point3 &orig,
                                    const Point3 &v) {
    HitRecord rec;
    if (!hit(Ray(orig, v), 0.001, FLT_MAX, rec))
      return 0;

    float area = (rect.a1 - rect.a0) * (rect.b1 - rect.b0);
    return get_pdf_surface(v, rec.normal, rec.t, area);
  }
  __device__ static Vec3 random(AaRect rect,
                                const Point3 &orig,
                                curandState *loc) override {
    Point3 random_point =
        Point3(random_float(loc, rect.a0, rect.a1), rect.k,
               random_float(loc, rect.b0, rect.b1));
    return random_point - orig;
  }
};

class XYRect : public AaRect {
public:
  float x0, x1, y0, y1;

public:
  __host__ __device__ XYRect() {}
  __host__ __device__ XYRect(float _x0, float _x1,
                             float _y0, float _y1, float _k,
                             MaterialParam mat)
      : AaRect(_x0, _x1, _y0, _y1, _k, mat, Vec3(0, 0, 1)),
        x0(_x0), x1(_x1), y0(_y0), y1(_y1) {}
};
class XZRect : public AaRect {
public:
  float x0, x1, z0, z1;

public:
  __host__ __device__ XZRect() {}
  __host__ __device__ XZRect(float _x0, float _x1,
                             float _z0, float _z1, float _k,
                             MaterialParam mat)
      : AaRect(_x0, _x1, _z0, _z1, _k, mat, Vec3(0, 1, 0)),
        x0(_x0), x1(_x1), z0(_z0), z1(_z1) {}
};
class YZRect : public AaRect {
public:
  float y0, y1, z0, z1;

public:
  __host__ __device__ YZRect() {}
  __host__ __device__ YZRect(float _y0, float _y1,
                             float _z0, float _z1, float _k,
                             MaterialParam mat)
      : AaRect(_y0, _y1, _z0, _z1, _k, mat, Vec3(1, 0, 0)),
        y0(_y0), y1(_y1), z0(_z0), z1(_z1) {}
};
