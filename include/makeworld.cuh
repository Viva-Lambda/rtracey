// make world kernel
#pragma once

#include <ray.cuh>
#include <scenegroup.cuh>
#include <scenematparam.cuh>
#include <sceneobj.cuh>
#include <sceneparam.cuh>
#include <sceneprim.cuh>
#include <scenetexparam.cuh>
#include <vec3.cuh>

SceneObjects make_cornell_box() {
  TextureParam red_solid =
      mkSolidColorParam(Color(.65, .05, .05));
  TextureParam green_solid =
      mkSolidColorParam(Color(.12, .45, .15));
  TextureParam blue_solid =
      mkSolidColorParam(Color(.05, .05, .65));
  TextureParam white_solid =
      mkSolidColorParam(Color(.75, .75, .75));
  TextureParam light_solid =
      mkSolidColorParam(Color(15, 15, 15));

  MaterialParam red_param = mkLambertParam(red_solid);
  MaterialParam green_param =
      mkMetalParam(green_solid, 0.3f);
  MaterialParam blue_param = mkLambertParam(blue_solid);
  MaterialParam white_param = mkLambertParam(white_solid);
  MaterialParam light_param =
      mkDiffuseLightParam(light_solid);

  HittableParam green_wall =
      mkYZRectHittable(0, 555, 0, 555, 555);
  auto red_wall = mkYZRectHittable(0, 555, 0, 555, 0);
  auto light_wall =
      mkXZRectHittable(213, 343, 227, 332, 554);
  auto white_wall = mkXZRectHittable(0, 555, 0, 555, 0);
  auto white_wall2 = mkXZRectHittable(0, 555, 0, 555, 555);
  auto blue_wall = mkXYRectHittable(0, 555, 0, 555, 555);
  //
  int prim_count = 6;
  ScenePrim *prms = new ScenePrim[prim_count];
  //
  int pcount = 0;
  prms[pcount] = ScenePrim(green_param, green_wall, 0, 0);
  pcount++;
  prms[pcount] = ScenePrim(red_param, red_wall, 1, 0);
  pcount++;
  prms[pcount] = ScenePrim(light_param, light_wall, 2, 0);
  pcount++;
  prms[pcount] = ScenePrim(white_param, white_wall, 3, 0);
  pcount++;
  prms[pcount] = ScenePrim(white_param, white_wall2, 4, 0);
  pcount++;
  prms[pcount] = ScenePrim(blue_param, blue_wall, 5, 0);
  TextureParam tp;
  SceneGroup sg(prms, prim_count, 0, SOLID, 0.0f, tp);
  SceneGroup sgs[] = {sg};
  SceneObjects sobjs(sgs, 1);
  // Hittables hs = sobjs.to_hittables();
  return sobjs;
}

void free_empty_cornell(
    thrust::device_ptr<Vec3> &fb,
    thrust::device_ptr<curandState> randState1,
    thrust::device_ptr<curandState> randState2) {

  thrust::device_free(fb);
  CUDA_CONTROL(cudaGetLastError());
  CUDA_CONTROL(cudaGetLastError());

  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState2);
  CUDA_CONTROL(cudaGetLastError());
  thrust::device_free(randState1);
  CUDA_CONTROL(cudaGetLastError());
}
