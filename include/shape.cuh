#pragma once
// shape
#include <matparam.cuh>
#include <record.cuh>
#include <vec3.cuh>

struct Triangle {
  const Point3 p1, p2, p3;
  __host__ __device__ Triangle()
      : p1(0.0f), p2(1.0f), p3(0.5f) {}
  __host__ __device__ Triangle(const Point3 &p_1,
                               const Point3 &p_2,
                               const Point3 &p_3)
      : p1(p_1), p2(p_2), p3(p_3) {}
};
struct AaTriangle : Triangle {
  Vec3 axis_normal;
  AxisInfo ax;
  Point3 p1, p2, p3;
  __host__ __device__ AaTriangle() {}
  __host__ __device__
  AaTriangle(const float &a0, const float &a1,
             const float &a2, const float &b0,
             const float &b1, const float &k, Vec3 anormal)
      : ax(AxisInfo(anormal)), axis_normal(anormal) {
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
struct XYTriangle : AaTriangle {

  __host__ __device__ XYTriangle() {}
  __host__ __device__ XYTriangle(
      const float &x0, const float &x1, const float &x2,
      const float &y0, const float &y1, const float &z)
      : AaTriangle(x0, x1, x2, y0, y1, z, Vec3(0, 0, 1)) {}
};

struct XZTriangle : AaTriangle {

  __host__ __device__ XZTriangle() {}
  __host__ __device__ XZTriangle(
      const float &x0, const float &x1, const float &x2,
      const float &z0, const float &z1, const float &y)
      : AaTriangle(x0, x1, x2, z0, z1, y, Vec3(0, 1, 0)) {}
};
struct YZTriangle : AaTriangle {
  __host__ __device__ YZTriangle() {}
  __host__ __device__ YZTriangle(
      const float &y0, const float &y1, const float &y2,
      const float &z0, const float &z1, const float &x)
      : AaTriangle(y0, y1, y2, z0, z1, x, Vec3(1, 0, 0)) {}
};
struct Sphere {
  __host__ __device__ Sphere()
      : center(0.0f), radius(0.0f) {}
  __host__ __device__ Sphere(const Point3 &cen,
                             const float &r)
      : center(cen.x(), cen.y(), cen.z()), radius(r){};
  const Vec3 center;
  const float radius;
};
struct MovingSphere {
  const Point3 center1, center2;
  const float time0, time1, radius;

  __host__ __device__ MovingSphere();
  __host__ __device__ MovingSphere(const Point3 &c1,
                                   const Point3 &c2,
                                   const float &t0,
                                   const float &t1,
                                   const float &rad)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad) {}
  __host__ __device__ Point3 center(float time) const {
    return center1 +
           (time - time0) / (time1 - time0) *
               (center2 - center1);
  }
  __host__ __device__ static Point3
  mcenter(Point3 c1, Point3 c2, float t1, float t2,
          float t) {
    return c1 + (t - t1) / (t2 - t1) * (c2 - c1);
  }
};
struct AaRect {
  const Vec3 axis_normal;
  const float a0, a1, // aligned1
      b0, b1;         // aligned2
  const AxisInfo ax;

  const float k;

  __host__ __device__ AaRect()
      : a0(0.0f), a1(1.0f), b0(0.0f), b1(1.0f), k(0.0f),
        axis_normal(Vec3(0.0f, 1.0f, 0.0f)),
        ax(AxisInfo(Vec3(0.0f, 1.0f, 0.0f))) {}
  __host__ __device__ AaRect(const float &a_0,
                             const float &a_1,
                             const float &b_0,
                             const float &b_1,
                             const float &_k,
                             const Vec3 &anormal)
      : a0(a_0), a1(a_1), b0(b_0), b1(b_1), k(_k),
        axis_normal(anormal), ax(AxisInfo(anormal)) {}
};
struct XYRect : public AaRect {
  const float x0, x1, y0, y1;

  __host__ __device__ XYRect()
      : x0(0.0f), x1(0.0f), y0(0.0f), y1(0.0f) {}
  __host__ __device__ XYRect(const float &_x0,
                             const float &_x1,
                             const float &_y0,
                             const float &_y1,
                             const float &_k)
      : AaRect(_x0, _x1, _y0, _y1, _k, Vec3(0, 0, 1)),
        x0(_x0), x1(_x1), y0(_y0), y1(_y1) {}
};
struct XZRect : public AaRect {
  const float x0, x1, z0, z1;

  __host__ __device__ XZRect()
      : x0(0.0f), x1(0.0f), z0(0.0f), z1(0.0f) {}
  __host__ __device__ XZRect(const float &_x0,
                             const float &_x1,
                             const float &_z0,
                             const float &_z1,
                             const float &_k)
      : AaRect(_x0, _x1, _z0, _z1, _k, Vec3(0, 1, 0)),
        x0(_x0), x1(_x1), z0(_z0), z1(_z1) {}
};
struct YZRect : public AaRect {
  const float y0, y1, z0, z1;

  __host__ __device__ YZRect()
      : y0(0.0f), y1(0.0f), z0(0.0f), z1(0.0f) {}
  __host__ __device__ YZRect(const float &_y0,
                             const float &_y1,
                             const float &_z0,
                             const float &_z1,
                             const float &_k)
      : AaRect(_y0, _y1, _z0, _z1, _k, Vec3(1, 0, 0)),
        y0(_y0), y1(_y1), z0(_z0), z1(_z1) {}
};
