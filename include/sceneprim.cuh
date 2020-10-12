#ifndef SCENEPRIM_CUH
#define SCENEPRIM_CUH

#include <material.cuh>
#include <ray.cuh>
#include <sceneaarect.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <scenesphere.cuh>
#include <scenetriangle.cuh>
#include <scenetype.cuh>
#include <texture.cuh>
#include <vec3.cuh>

struct ScenePrim {
  // material params
  MaterialParam mparam;

  // hittable params
  HittableParam hparam;

  // group params
  int group_id;
  int group_index;

  __host__ __device__ ScenePrim() {}
  __host__ __device__ ScenePrim(MaterialParam mt,
                                HittableParam ht,
                                int gindex, int gid)
      : mparam(mt), hparam(ht), group_index(gindex),
        group_id(gid) {}
  __host__ __device__ Sphere to_sphere() {
    Point3 cent(hparam.p1x, hparam.p1y, hparam.p1z);
    Sphere sp(cent, hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ MovingSphere to_moving_sphere() {
    Point3 cent1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 cent2(hparam.p2x, hparam.p2y, hparam.p2z);
    MovingSphere sp(cent1, cent2, hparam.n1x, hparam.n1y,
                    hparam.radius, mparam);
    return sp;
  }
  __host__ __device__ Triangle to_triangle() {
    Point3 p1(hparam.p1x, hparam.p1y, hparam.p1z);
    Point3 p2(hparam.p2x, hparam.p2y, hparam.p2z);
    Point3 p3(hparam.n1x, hparam.n1y, hparam.n1z);
    Triangle tri(p1, p2, p3, mparam);
    return tri;
  }
  __host__ __device__ void rect_val(float &a0, float &a1,
                                    float &b0, float &b1,
                                    float &k) {
    a0 = hparam.p1x;
    a1 = hparam.p1y;
    b0 = hparam.p2x;
    b1 = hparam.p2y;
    k = hparam.radius;
  }
  __host__ __device__ XYRect to_xyrect() {
    float x0, x1, y0, y1, k;
    rect_val(x0, x1, y0, y1, k);
    XYRect xyr(x0, x1, y0, y1, k, mparam);
    return xyr;
  }
  __host__ __device__ XZRect to_xzrect() {
    float x0, x1, z0, z1, k;
    rect_val(x0, x1, z0, z1, k);
    XZRect xzr(x0, x1, z0, z1, k, mparam);
    return xzr;
  }
  __host__ __device__ YZRect to_yzrect() {
    float y0, y1, z0, z1, k;
    rect_val(y0, y1, z0, z1, k);
    YZRect yzr(y0, y1, z0, z1, k, mparam);
    return yzr;
  }
};

template <> struct SceneHittable<ScenePrim> {
  __device__ static bool hit(ScenePrim sprim, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec) {
    bool res = false;
    switch (sprim.hparam.htype) {
    case XY_RECT: {
      XYRect xyr = sprim.to_xyrect();
      res = SceneHittable<XYRect>::hit(xyr, r, d_min, d_max,
                                       rec);
      break;
    }
    case YZ_RECT: {
      YZRect yzr = sprim.to_yzrect();
      res = SceneHittable<YZRect>::hit(yzr, r, d_min, d_max,
                                       rec);
      break;
    }
    case XZ_RECT: {
      XZRect xzr = sprim.to_xzrect();
      res = SceneHittable<XZRect>::hit(xzr, r, d_min, d_max,
                                       rec);
      break;
    }
    case XY_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::hit(tri, r, d_min,
                                         d_max, rec);
      break;
    }
    case XZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::hit(tri, r, d_min,
                                         d_max, rec);
      break;
    }
    case YZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::hit(tri, r, d_min,
                                         d_max, rec);
      break;
    }
    case SPHERE: {
      Sphere sp = sprim.to_sphere();

      res = SceneHittable<Sphere>::hit(sp, r, d_min, d_max,
                                       rec);
      break;
    }
    case MOVING_SPHERE: {
      MovingSphere sp = sprim.to_moving_sphere();
      res = SceneHittable<MovingSphere>::hit(sp, r, d_min,
                                             d_max, rec);
      break;
    }
    }
    return res;
  }

  __host__ __device__ static bool
  bounding_box(ScenePrim sprim, float t0, float t1,
               Aabb &output_box) {
    bool res = false;
    switch (sprim.hparam.htype) {
    case XY_RECT: {
      XYRect xyr = sprim.to_xyrect();
      res = SceneHittable<XYRect>::bounding_box(xyr, t0, t1,
                                                output_box);
      break;
    }
    case YZ_RECT: {
      YZRect yzr = sprim.to_yzrect();
      res = SceneHittable<YZRect>::bounding_box(yzr, t0, t1,
                                                output_box);
      break;
    }
    case XZ_RECT: {
      XZRect xzr = sprim.to_xzrect();
      res = SceneHittable<XZRect>::bounding_box(xzr, t0, t1,
                                                output_box);
      break;
    }
    case XY_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::bounding_box(
          tri, t0, t1, output_box);
      break;
    }
    case XZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::bounding_box(
          tri, t0, t1, output_box);
      break;
    }
    case YZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      res = SceneHittable<Triangle>::bounding_box(
          tri, t0, t1, output_box);
      break;
    }
    case SPHERE: {
      Sphere sp = sprim.to_sphere();

      res = SceneHittable<Sphere>::bounding_box(sp, t0, t1,
                                                output_box);
      break;
    }
    case MOVING_SPHERE: {
      MovingSphere sp = sprim.to_moving_sphere();
      res = SceneHittable<MovingSphere>::bounding_box(
          sp, t0, t1, output_box);
      break;
    }
    }
    return res;
  }

  __device__ static float pdf_value(ScenePrim sprim,
                                    const Point3 &o,
                                    const Point3 &v) {
    float pdf = 1.0f;

    switch (sprim.hparam.htype) {
    case XY_RECT: {
      XYRect xyr = sprim.to_xyrect();
      pdf = SceneHittable<XYRect>::pdf_value(xyr, o, v);
      break;
    }
    case YZ_RECT: {
      YZRect yzr = sprim.to_yzrect();
      pdf = SceneHittable<YZRect>::pdf_value(yzr, o, v);
      break;
    }
    case XZ_RECT: {
      XZRect xzr = sprim.to_xzrect();
      pdf = SceneHittable<XZRect>::pdf_value(xzr, o, v);
      break;
    }
    case XY_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      pdf = SceneHittable<Triangle>::pdf_value(tri, o, v);
      break;
    }
    case XZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      pdf = SceneHittable<Triangle>::pdf_value(tri, o, v);
      break;
    }
    case YZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      pdf = SceneHittable<Triangle>::pdf_value(tri, o, v);
      break;
    }
    case SPHERE: {
      Sphere sp = sprim.to_sphere();

      pdf = SceneHittable<Sphere>::pdf_value(sp, o, v);
      break;
    }
    case MOVING_SPHERE: {
      MovingSphere sp = sprim.to_moving_sphere();
      pdf =
          SceneHittable<MovingSphere>::pdf_value(sp, o, v);
      break;
    }
    }
    return pdf;
  }
  __device__ static Vec3
  random(ScenePrim sprim, const Vec3 &v, curandState *loc) {
    Vec3 vp(0.0f);
    switch (sprim.hparam.htype) {
    case XY_RECT: {
      XYRect xyr = sprim.to_xyrect();
      vp = SceneHittable<XYRect>::random(xyr, v, loc);
      break;
    }
    case YZ_RECT: {
      YZRect yzr = sprim.to_yzrect();
      vp = SceneHittable<YZRect>::random(yzr, v, loc);
      break;
    }
    case XZ_RECT: {
      XZRect xzr = sprim.to_xzrect();
      vp = SceneHittable<XZRect>::random(xzr, v, loc);
      break;
    }
    case XY_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      vp = SceneHittable<Triangle>::random(tri, v, loc);
      break;
    }
    case XZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      vp = SceneHittable<Triangle>::random(tri, v, loc);
      break;
    }
    case YZ_TRIANGLE: {
      Triangle tri = sprim.to_triangle();
      vp = SceneHittable<Triangle>::random(tri, v, loc);
      break;
    }
    case SPHERE: {
      Sphere sp = sprim.to_sphere();
      vp = SceneHittable<Sphere>::random(sp, v, loc);
      break;
    }
    case MOVING_SPHERE: {
      MovingSphere sp = sprim.to_moving_sphere();
      vp = SceneHittable<MovingSphere>::random(sp, v, loc);
      break;
    }
    }
    return vp;
  }
};
#endif
