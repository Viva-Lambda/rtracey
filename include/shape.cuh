#pragma once
// shape
#include <matparam.cuh>
#include <record.cuh>
#include <vec3.cuh>

struct Triangle {
  Point3 p1, p2, p3;
  MaterialParam mat_ptr;
  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(Point3 p_1, Point3 p_2,
                               Point3 p_3,
                               const MaterialParam &mpram)
      : p1(p_1), p2(p_2), p3(p_3), mat_ptr(mpram) {}
};
struct AaTriangle : Triangle {
  Vec3 axis_normal;
  AxisInfo ax;
  MaterialParam mat_ptr;
  Point3 p1, p2, p3;
  __host__ __device__ AaTriangle() {}
  __host__ __device__ AaTriangle(float *a0, float *a1,
                                 float *a2, float *b0,
                                 float *b1, float *k,
                                 const MaterialParam &mp,
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
struct XYTriangle : AaTriangle {

  __host__ __device__ XYTriangle() {}
  __host__ __device__ XYTriangle(float *x0, float *x1,
                                 float *x2, float *y0,
                                 float *y1, float *z,
                                 const MaterialParam &mp)
      : AaTriangle(x0, x1, x2, y0, y1, z, mp,
                   Vec3(0, 0, 1)) {}
};
struct XZTriangle : AaTriangle {

  __host__ __device__ XZTriangle() {}
  __host__ __device__ XZTriangle(float *x0, float *x1,
                                 float *x2, float *z0,
                                 float *z1, float *y,
                                 const MaterialParam &mp)
      : AaTriangle(x0, x1, x2, z0, z1, y, mp,
                   Vec3(0, 1, 0)) {}
};
struct YZTriangle : AaTriangle {
  __host__ __device__ YZTriangle() {}
  __host__ __device__ YZTriangle(float *y0, float *y1,
                                 float *y2, float *z0,
                                 float *z1, float *x,
                                 const MaterialParam &mp)
      : AaTriangle(y0, y1, y2, z0, z1, x, mp,
                   Vec3(1, 0, 0)) {}
};
struct Sphere {
  __host__ __device__ Sphere() {}
  __host__ __device__ Sphere(const Point3 &cen, float *r,
                             const MaterialParam &mat_ptr_)
      : center(cen.e1, cen.e2, cen.e3), radius(r),
        mat_ptr(mat_ptr_){};
  Vec3 center;
  float *radius;
  MaterialParam mat_ptr;
};
struct MovingSphere {
  Point3 center1, center2;
  float *time0, *time1, *radius;
  MaterialParam mat_ptr;

  __host__ __device__ MovingSphere();
  __host__ __device__ MovingSphere(const Point3 &c1,
                                   const Point3 &c2,
                                   float *t0, float *t1,
                                   float *rad,
                                   const MaterialParam &mat)
      : center1(c1), center2(c2), time0(t0), time1(t1),
        radius(rad), mat_ptr(mat) {}
  __host__ __device__ Point3 center(float *time) const {
    return center1 +
           ((*time - (*time0)) / (*time1 - (*time0))) *
               (center2 - center1);
  }
};
struct AaRect {
  Vec3 axis_normal;
  float *a0, *a1, // aligned1
      *b0, *b1;   // aligned2
  AxisInfo ax;

  float *k;
  MaterialParam mat_ptr;

  __host__ __device__ AaRect() {}
  __host__ __device__ AaRect(float *a_0, float *a_1,
                             float *b_0, float *b_1,
                             float *_k,
                             const MaterialParam &mat,
                             const Vec3 &anormal)
      : a0(a_0), a1(a_1), b0(b_0), b1(b_1), k(_k),
        mat_ptr(mat), axis_normal(anormal) {
    ax = AxisInfo(axis_normal);
  }
};
struct XYRect : public AaRect {
  float *x0, *x1, *y0, *y1;

  __host__ __device__ XYRect() {}
  __host__ __device__ XYRect(float *_x0, float *_x1,
                             float *_y0, float *_y1,
                             float *_k,
                             const MaterialParam &mat)
      : AaRect(_x0, _x1, _y0, _y1, _k, mat, Vec3(0, 0, 1)),
        x0(_x0), x1(_x1), y0(_y0), y1(_y1) {}
};
struct XZRect : public AaRect {
  float *x0, *x1, *z0, *z1;

  __host__ __device__ XZRect() {}
  __host__ __device__ XZRect(float *_x0, float *_x1,
                             float *_z0, float *_z1,
                             float *_k,
                             const MaterialParam &mat)
      : AaRect(_x0, _x1, _z0, _z1, _k, mat, Vec3(0, 1, 0)),
        x0(_x0), x1(_x1), z0(_z0), z1(_z1) {}
};
struct YZRect : public AaRect {
  float *y0, *y1, *z0, *z1;

  __host__ __device__ YZRect() {}
  __host__ __device__ YZRect(float *_y0, float *_y1,
                             float *_z0, float *_z1,
                             float *_k,
                             const MaterialParam &mat)
      : AaRect(_y0, _y1, _z0, _z1, _k, mat, Vec3(1, 0, 0)),
        y0(_y0), y1(_y1), z0(_z0), z1(_z1) {}
};
