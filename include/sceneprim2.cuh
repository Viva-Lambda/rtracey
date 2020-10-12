#ifndef SCENEPRIM_CUH
#define SCENEPRIM_CUH

#include <aarect.cuh>
#include <hittable.cuh>
#include <material.cuh>
#include <ray.cuh>
#include <scenehit.cuh>
#include <sceneparam.cuh>
#include <scenetype.cuh>
#include <sphere.cuh>
#include <texture.cuh>
#include <triangle.cuh>
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
  template <HittableType T>
  __device__ static bool hit(ScenePrim sprim, const Ray &r,
                             float d_min, float d_max,
                             HitRecord &rec);
  template <HittableType T>
  __host__ __device__ static bool
  bounding_box(ScenePrim sprim, float t0, float t1,
               Aabb &output_box);

  template <HittableType T>
  __device__ static float pdf_value(ScenePrim sprim,
                                    const Point3 &o,
                                    const Point3 &v);
  template <HittableType T>
  __device__ static Vec3
  random(ScenePrim sprim, const Vec3 &v, curandState *loc);
};

template <>
__device__ bool SceneHittable<ScenePrim>::hit<XY_RECT>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  XYRect xyr = sprim.to_xyrect();
  return SceneHittable<XYRect>::hit(xyr, r, d_min, d_max,
                                    rec);
}
template <>
__device__ bool SceneHittable<ScenePrim>::hit<XZ_RECT>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  XZRect xzr = sprim.to_xzrect();
  return SceneHittable<XZRect>::hit(xzr, r, d_min, d_max,
                                    rec);
}
template <>
__device__ bool SceneHittable<ScenePrim>::hit<YZ_RECT>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  YZRect yzr = sprim.to_yzrect();
  return SceneHittable<YZRect>::hit(yzr, r, d_min, d_max,
                                    rec);
}
template <>
__device__ bool SceneHittable<ScenePrim>::hit<SPHERE>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  Sphere sp = sprim.to_sphere();
  return SceneHittable<Sphere>::hit(sp, r, d_min, d_max,
                                    rec);
}
template <>
__device__ bool SceneHittable<ScenePrim>::hit<TRIANGLE>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  Triangle tri = sprim.to_triangle();
  return SceneHittable<Triangle>::hit(tri, r, d_min, d_max,
                                      rec);
}

template <>
__device__ bool
SceneHittable<ScenePrim>::hit<MOVING_SPHERE>(
    ScenePrim sprim, const Ray &r, float d_min, float d_max,
    HitRecord &rec) {
  MovingSphere msp = sprim.to_moving_sphere();
  return SceneHittable<MovingSphere>::hit(msp, r, d_min,
                                          d_max, rec);
}

template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<XY_RECT>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  XYRect xyr = sprim.to_xyrect();
  return SceneHittable<XYRect>::bounding_box(xyr, t0, t1,
                                             output_box);
}
template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<XZ_RECT>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  XZRect xyr = sprim.to_xzrect();
  return SceneHittable<XZRect>::bounding_box(xyr, t0, t1,
                                             output_box);
}
template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<YZ_RECT>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  YZRect xyr = sprim.to_yzrect();
  return SceneHittable<YZRect>::bounding_box(xyr, t0, t1,
                                             output_box);
}
template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<SPHERE>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  Sphere xyr = sprim.to_sphere();
  return SceneHittable<Sphere>::bounding_box(xyr, t0, t1,
                                             output_box);
}
template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<TRIANGLE>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  Triangle xyr = sprim.to_triangle();
  return SceneHittable<Triangle>::bounding_box(xyr, t0, t1,
                                               output_box);
}
template <>
__host__ __device__ bool
SceneHittable<ScenePrim>::bounding_box<MOVING_SPHERE>(
    ScenePrim sprim, float t0, float t1, Aabb &output_box) {
  MovingSphere xyr = sprim.to_moving_sphere();
  return SceneHittable<MovingSphere>::bounding_box(
      xyr, t0, t1, output_box);
}

template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<XY_RECT>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  XYRect xyr = sprim.to_xyrect();
  return SceneHittable<XYRect>::pdf_value(xyr, o, v);
}
template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<XZ_RECT>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  XZRect xyr = sprim.to_xzrect();
  return SceneHittable<XZRect>::pdf_value(xyr, o, v);
}
template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<YZ_RECT>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  YZRect xyr = sprim.to_yzrect();
  return SceneHittable<YZRect>::pdf_value(xyr, o, v);
}
template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<SPHERE>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  Sphere xyr = sprim.to_sphere();
  return SceneHittable<Sphere>::pdf_value(xyr, o, v);
}
template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<TRIANGLE>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  Triangle xyr = sprim.to_triangle();
  return SceneHittable<Triangle>::pdf_value(xyr, o, v);
}
template <>
__host__ __device__ float
SceneHittable<ScenePrim>::pdf_value<MOVING_SPHERE>(
    ScenePrim sprim, const Point3 &o, const Point3 &v) {
  MovingSphere xyr = sprim.to_moving_sphere();
  return SceneHittable<MovingSphere>::pdf_value(xyr, o, v);
}

template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<XY_RECT>(
    ScenePrim sprim, const Vec3 &v, curandState *loc) {
  XYRect xyr = sprim.to_xyrect();
  return SceneHittable<XYRect>::random(xyr, v, loc);
}
template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<XZ_RECT>(
    ScenePrim sprim, const Vec3 &v, curandState *loc) {
  XZRect xyr = sprim.to_xzrect();
  return SceneHittable<XZRect>::random(xyr, v, loc);
}
template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<YZ_RECT>(
    ScenePrim sprim, const Vec3 &v, curandState *loc) {
  YZRect xyr = sprim.to_yzrect();
  return SceneHittable<YZRect>::random(xyr, v, loc);
}
template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<SPHERE>(ScenePrim sprim,
                                         const Vec3 &v,
                                         curandState *loc) {
  Sphere xyr = sprim.to_sphere();
  return SceneHittable<Sphere>::random(xyr, v, loc);
}
template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<TRIANGLE>(
    ScenePrim sprim, const Vec3 &v, curandState *loc) {
  Triangle xyr = sprim.to_triangle();
  return SceneHittable<Triangle>::random(xyr, v, loc);
}
template <>
__host__ __device__ Vec3
SceneHittable<ScenePrim>::random<MOVING_SPHERE>(
    ScenePrim sprim, const Vec3 &v, curandState *loc) {
  MovingSphere xyr = sprim.to_moving_sphere();
  return SceneHittable<MovingSphere>::random(xyr, v, loc);
}

#endif
