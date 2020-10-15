#pragma once

#include <camera.cuh>
#include <external.hpp>
#include <pdf.cuh>
#include <ray.cuh>
#include <scenegroup.cuh>
#include <sceneobj.cuh>
#include <vec3.cuh>

/**
  @param Ray r is the incoming ray.
  @param Hittables** world pointer to list of hittables
 */
__device__ Color ray_color(const Ray &r,
                           const SceneObjects world,
                           curandState *loc, int bounceNb) {
  Ray current_ray = r;
  Vec3 current_attenuation = Vec3(1.0f);
  Vec3 result = Vec3(0.0f);

  while (bounceNb > 0) {
    HitRecord rec;
    bool anyHit =
        world.hit(current_ray, 0.001f, FLT_MAX, rec);
    if (anyHit) {
      // rec.mat_ptr.tparam.tdata = world.tdata;
      Color emittedColor = world.emitted(rec);
      Ray scattered;
      Vec3 attenuation;
      float pdf_val = 1.0f;
      bool isScattered =
          world.scatter(current_ray, rec, attenuation,
                        scattered, pdf_val);
      if (isScattered) {
        bounceNb--;
        float s_pdf =
            SceneMaterial<MaterialParam>::scattering_pdf(
                rec.mat_ptr, current_ray, rec, scattered);
        result += (current_attenuation * emittedColor);
        current_attenuation *=
            attenuation * s_pdf / pdf_val;
        current_ray = scattered;
      } else {
        result += (current_attenuation * emittedColor);
        return result;
      }
    } else {
      return Color(0.0);
    }
  }
  // return Vec3(1.0f); // background color
  return result;
}
__global__ void render(Vec3 *fb, int maximum_x,
                       int maximum_y, int sample_nb,
                       int bounceNb, Camera dcam,
                       const SceneObjects world,
                       curandState *randState) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= maximum_x) || (j >= maximum_y)) {
    return;
  }
  int pixel_index = j * maximum_x + i;
  curandState localS = randState[pixel_index];
  Vec3 rcolor(0.0f);
  Camera cam = dcam;
  // world.set_rand(&localS);
  for (int s = 0; s < sample_nb; s++) {
    float u = float(i + curand_uniform(&localS)) /
              float(maximum_x);
    float v = float(j + curand_uniform(&localS)) /
              float(maximum_y);
    Ray r = cam.get_ray(u, v, &localS);
    //
    rcolor += ray_color(r, world, &localS, bounceNb);
    // rcolor += Color(world.tp1xs[0], world.tp1ys[0],
    //                world.tp1zs[0]);
  }
  // fix the bounce depth
  randState[pixel_index] = localS;
  rcolor /= float(sample_nb);
  rcolor.e[0] = sqrt(rcolor.x());
  rcolor.e[1] = sqrt(rcolor.y());
  rcolor.e[2] = sqrt(rcolor.z());
  fb[pixel_index] = rcolor;
}
