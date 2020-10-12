#pragma once
#include <onb.cuh>
#include <ray.cuh>
#include <scenemat.cuh>
#include <sceneparam.cuh>
#include <texture.cuh>
#include <vec3.cuh>

template <> struct SceneMaterial<Lambertian> {
  __device__ static bool
  scatter(Lambertian lamb, const Ray &r_in,
          const HitRecord &rec, Color &attenuation,
          Ray &scattered, float &pdf, curandState *loc) {
    Onb uvw;
    uvw.build_from_w(rec.normal);
    auto direction =
        uvw.local(random_cosine_direction(loc));
    scattered = Ray(rec.p, to_unit(direction), r_in.time());
    attenuation = lamb.albedo.value(rec.u, rec.v, rec.p);
    pdf = dot(uvw.w(), scattered.direction()) / M_PI;
    return true;
  }
  __host__ __device__ static float
  scattering_pdf(Lambertian lamb, const Ray &r_in,
                 const HitRecord &rec,
                 const Ray &scattered) override {
    auto cosine =
        dot(rec.normal, to_unit(scattered.direction()));
    return cosine < 0 ? 0 : cosine / M_PI;
  }
};
