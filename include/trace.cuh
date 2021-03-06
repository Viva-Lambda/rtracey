#pragma once

#include <camera.cuh>
#include <external.hpp>
#include <hit.cuh>
#include <pdf.cuh>
#include <prandom.cuh>
#include <ray.cuh>
#include <sceneobj.cuh>
#include <shade.cuh>
#include <spdf.cuh>
#include <vec3.cuh>

/**
  @param Ray r is the incoming ray.
  @param Hittables** world pointer to list of hittables
 */
__device__ Color ray_color(const Ray &r,
                           const SceneObjects &world,
                           curandState *loc, int bounceNb) {
  Ray current_ray = r;
  Vec3 current_attenuation = Vec3(1.0f, 1.0f, 1.0f);
  Vec3 result = Vec3(0.0f, 0.0f, 0.0f);
  bool is_gs[] = {true, false};
  int inds[] = {1, 2};
  ScatterRecord srec;
  // delete[] is_gs;
  // delete[] inds;
  while (bounceNb > 0) {
    HitRecord rec;
    bool anyHit = hit<SCENE>(world, current_ray, 0.001f,
                             FLT_MAX, rec, loc);
    if (!anyHit) {
      return Color(0.0f);
    }

    Color emittedColor = emitted<MATERIAL>(world, rec, loc);
    bool isScattered = scatter<MATERIAL>(world, current_ray,
                                         rec, srec, loc);
    if (!isScattered) {
      // object does not scatter the light
      // most probably object is a light source

      result += (current_attenuation * emittedColor);
      return result;
    }
    bounceNb--;
    // object scatters the light
    if (srec.is_specular) {
      // object is specular
      current_attenuation *= srec.attenuation;
      current_ray = srec.specular_ray;
    } else {
      // object is not specular
      // float s_pdf = scattering_pdf<MATERIAL>(
      //    world, current_ray, rec, scattered);
      // pdf value
      float pdf_val = pdf_value<PDF>(world, rec, srec);
      Ray r_out = Ray(
          rec.p, pdf_generate<PDF>(world, rec, srec, loc),
          current_ray.time());
      float s_pdf = scattering_pdf<MATERIAL>(
          world, current_ray, rec, r_out);

      // scattered ray
      current_ray = r_out;

      // attenuation
      Color attenuation = srec.attenuation;

      result += (current_attenuation * emittedColor);
      current_attenuation *= attenuation * s_pdf / pdf_val;
      current_ray = srec.specular_ray;
    }
  }
  return Color(0.0f); // background color
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
  Vec3 rcolor(0.0f, 0.0f, 0.0f);
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
  fb[pixel_index] = sqrt(rcolor);
}

/**
  @param Ray r is the incoming ray.
  @param Hittables** world pointer to list of hittables
 */
__host__ Color h_ray_color(const Ray &r,
                           const SceneObjects &world,
                           int bounceNb) {
  Ray current_ray = r;
  Vec3 current_attenuation = Vec3(1.0f, 1.0f, 1.0f);
  Vec3 result = Vec3(0.0f, 0.0f, 0.0f);
  // delete[] is_gs;
  // delete[] inds;
  while (bounceNb > 0) {
    HitRecord rec;
    bool anyHit = h_hit<SCENE>(world, current_ray, 0.001f,
                               FLT_MAX, rec);
    if (!anyHit) {
      return Color(0.0f);
    }
    ScatterRecord srec;

    Color emittedColor = h_emitted<MATERIAL>(world, rec);
    bool isScattered =
        h_scatter<MATERIAL>(world, current_ray, rec, srec);
    if (!isScattered) {
      // object does not scatter the light
      // most probably object is a light source

      result += (current_attenuation * emittedColor);
      return result;
    }
    bounceNb--;
    // object scatters the light
    if (srec.is_specular) {
      // object is specular
      current_attenuation *= srec.attenuation;
      current_ray = srec.specular_ray;
    } else {
      // object is not specular
      // float s_pdf = scattering_pdf<MATERIAL>(
      //    world, current_ray, rec, scattered);
      // pdf value
      float pdf_val = pdf_value<PDF>(world, rec, srec);
      Ray r_out =
          Ray(rec.p, h_pdf_generate<PDF>(world, rec, srec),
              current_ray.time());
      float s_pdf = scattering_pdf<MATERIAL>(
          world, current_ray, rec, r_out);

      // scattered ray
      current_ray = r_out;

      // attenuation
      Color attenuation = srec.attenuation;

      result += (current_attenuation * emittedColor);
      current_attenuation *= attenuation * s_pdf / pdf_val;
      current_ray = srec.specular_ray;
    }
  }
  return Color(0.0f); // background color
}
__host__ void h_render(int WIDTH, int HEIGHT, int sample_nb,
                       int bounceNb, Camera dcam,
                       const SceneObjects &world) {
  for (int j = HEIGHT - 1; j >= 0; j--) {
    for (int i = 0; i < WIDTH; i++) {
      Vec3 pixel(0.0f, 0.0f, 0.0f);
      Camera cam = dcam;
      // world.set_rand(&localS);
      for (int s = 0; s < sample_nb; s++) {
        float u = float(i + hrandf()) / float(WIDTH);
        float v = float(j + hrandf()) / float(HEIGHT);
        Ray r = cam.h_get_ray(u, v);
        //
        pixel += h_ray_color(r, world, bounceNb);
      }
      pixel /= float(sample_nb);
      pixel = sqrt(pixel);

      int ir = int(255.99 * pixel.r());
      int ig = int(255.99 * pixel.g());
      int ib = int(255.99 * pixel.b());

      std::cout << ir << " " << ig << " " << ib
                << std::endl;
    }
  }
}
