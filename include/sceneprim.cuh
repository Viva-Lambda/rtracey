#pragma once
#include <aabb.cuh>
#include <primitive.cuh>
#include <ray.cuh>
#include <scenematparam.cuh>
#include <sceneshape.cuh>
#include <scenetype.cuh>
#include <shape.cuh>
#include <vec3.cuh>

template <> struct SceneHittable<Primitive> {
  __device__ static bool hit(const Primitive &sprim,
                             const Ray &r, float d_min,
                             float d_max, HitRecord &rec) {
    bool res = false;
    switch (sprim.hparam.htype) {
    case NONE_HITTABLE:
      res = false;
      break;
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
  bounding_box(const Primitive &sprim, float t0, float t1,
               Aabb &output_box) {
    bool res = false;
    switch (sprim.hparam.htype) {
    case NONE_HITTABLE:
      res = false;
      break;
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

  __device__ static float pdf_value(const Primitive &sprim,
                                    const Point3 &o,
                                    const Point3 &v) {
    float pdf = 1.0f;

    switch (sprim.hparam.htype) {
    case NONE_HITTABLE:
      pdf = 1.0f;
      break;
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
  __device__ static Vec3 random(const Primitive &sprim,
                                const Vec3 &v,
                                curandState *loc) {
    Vec3 vp(0.0f);
    switch (sprim.hparam.htype) {
    case NONE_HITTABLE:
      vp = Vec3(0.0f);
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

template <> struct SceneMaterial<Primitive> {
  __device__ static bool
  scatter(const Primitive &m, const Ray &r_in,
          const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    return SceneMaterial<MaterialParam>::scatter(
        m.mparam, r_in, rec, attenuation, scattered, pdf,
        loc);
  }
  __device__ static float
  scattering_pdf(const Primitive &m, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) {
    //
    return SceneMaterial<MaterialParam>::scattering_pdf(
        m.mparam, r_in, rec, scattered);
  }
  __device__ static Color emitted(const Primitive &m,
                                  float u, float v,
                                  const Point3 &p) {
    return SceneMaterial<MaterialParam>::emitted(m.mparam,
                                                 u, v, p);
  }
};
