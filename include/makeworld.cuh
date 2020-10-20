// make world kernel
#pragma once

#include <groupparam.cuh>
#include <matparam.cuh>
#include <primitive.cuh>
#include <ray.cuh>
#include <sceneobj.cuh>
#include <texparam.cuh>
#include <vec3.cuh>

SceneObjects make_cornell_box() {
  const TextureParam red_solid =
      mkSolidColorParam(Color(.65, .05, .05));
  const TextureParam green_solid =
      mkSolidColorParam(Color(.12, .75, .15));
  const TextureParam blue_solid =
      mkSolidColorParam(Color(.05, .05, .65));
  const TextureParam white_solid =
      mkSolidColorParam(Color(.75, .75, .75));
  const TextureParam light_solid =
      mkSolidColorParam(Color(25, 25, 25));

  const float fzz3 = 0.2f;
  const MaterialParam red_param =
      mkMetalParam(red_solid, fzz3);
  //
  const float fzz = 0.3f;
  const MaterialParam green_param =
      mkMetalParam(green_solid, fzz);
  //
  const float fzz2 = 0.1f;
  const MaterialParam blue_param =
      mkMetalParam(blue_solid, fzz2);
  //
  const MaterialParam white_param =
      mkLambertParam(white_solid);
  //
  const MaterialParam light_param =
      mkDiffuseLightParam(light_solid);
  //
  const HittableParam green_wall =
      mkYZRectHittable(0, 555, 0, 555, 555);
  const auto red_wall = mkYZRectHittable(0, 555, 0, 555, 0);
  const auto light_wall =
      mkXZRectHittable(213, 343, 227, 332, 554);
  const auto white_wall =
      mkXZRectHittable(0, 555, 0, 555, 0);
  const auto white_wall2 =
      mkXZRectHittable(0, 555, 0, 555, 555);
  const auto blue_wall =
      mkXYRectHittable(0, 555, 0, 555, 555);
  //
  const int prim_count = 6;

  int pcount0 = 0;
  int pcount1 = 1;
  int pcount2 = 2;
  int pcount3 = 3;
  int pcount4 = 4;
  int pcount5 = 5;
  const int group_id = 0;
  Primitive ps[] = {
      Primitive(green_param, green_wall, pcount0, group_id),
      Primitive(red_param, red_wall, pcount1, group_id),
      Primitive(light_param, light_wall, pcount2, group_id),
      Primitive(white_param, white_wall, pcount3, group_id),
      Primitive(white_param, white_wall2, pcount4,
                group_id),
      Primitive(blue_param, blue_wall, pcount5, group_id)};
  const TextureParam tp;
  const float g_dens = 0.0f;
  GroupParam sg(ps, prim_count, group_id, BOX, g_dens, tp);

  // first box

  GroupParam box1 =
      makeBox(Point3(130, 0, 65), Point3(295, 165, 230),
              white_param, 1);

  // second box
  GroupParam box2 =
      makeBox(Point3(265, 0, 295), Point3(430, 330, 460),
              white_param, 2);
  //

  GroupParam *sgs = new GroupParam[3];
  sgs[0] = sg;
  sgs[1] = box1;
  sgs[2] = box2;
  SceneObjects sobjs(sgs, 3);
  // box1.g_free();
  // box2.g_free();
  // sg.g_free();
  // Hittables hs = sobjs.to_hittables();
  return sobjs;
}
__global__ void make_cornell_box_k(SceneObjects world,
                                   curandState *loc) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const TextureParam red_solid =
        mkSolidColorParam(Color(.65, .05, .05));
    const TextureParam green_solid =
        mkSolidColorParam(Color(.12, .45, .15));
    const TextureParam blue_solid =
        mkSolidColorParam(Color(.05, .05, .65));
    const TextureParam white_solid =
        mkSolidColorParam(Color(.75, .75, .75));
    const TextureParam light_solid =
        mkSolidColorParam(Color(15, 15, 15));

    const MaterialParam red_param =
        mkLambertParam(red_solid);
    //
    const float fzz = 0.6f;
    const MaterialParam green_param =
        mkMetalParam(green_solid, fzz);
    //
    const MaterialParam blue_param =
        mkLambertParam(blue_solid);
    //
    const MaterialParam white_param =
        mkLambertParam(white_solid);
    //
    const MaterialParam light_param =
        mkDiffuseLightParam(light_solid);
    //
    const HittableParam green_wall =
        mkYZRectHittable(0, 555, 0, 555, 555);
    const auto red_wall =
        mkYZRectHittable(0, 555, 0, 555, 0);
    const auto light_wall =
        mkXZRectHittable(213, 343, 227, 332, 554);
    const auto white_wall =
        mkXZRectHittable(0, 555, 0, 555, 0);
    const auto white_wall2 =
        mkXZRectHittable(0, 555, 0, 555, 555);
    const auto blue_wall =
        mkXYRectHittable(0, 555, 0, 555, 555);
    //
    const int prim_count = 6;

    int pcount0 = 0;
    int pcount1 = 1;
    int pcount2 = 2;
    int pcount3 = 3;
    int pcount4 = 4;
    int pcount5 = 5;
    const int group_id = 0;
    Primitive ps[] = {
        Primitive(green_param, green_wall, pcount0,
                  group_id),
        Primitive(red_param, red_wall, pcount1, group_id),
        Primitive(light_param, light_wall, pcount2,
                  group_id),
        Primitive(white_param, white_wall, pcount3,
                  group_id),
        Primitive(white_param, white_wall2, pcount4,
                  group_id),
        Primitive(blue_param, blue_wall, pcount5,
                  group_id)};
    const TextureParam tp;
    const float g_dens = 0.0f;
    GroupParam sg(ps, prim_count, group_id, BOX, g_dens,
                  tp);

    // first box
    GroupParam box1 =
        makeBox(Point3(130, 0, 65), Point3(295, 165, 230),
                white_param, 1);

    // second box
    GroupParam box2 =
        makeBox(Point3(265, 0, 295), Point3(430, 330, 460),
                white_param, 2);
    //

    GroupParam sgs[] = {sg, box1, box2};
    SceneObjects sobjs(sgs, 3, loc);
    // Hittables hs = sobjs.to_hittables();
    // Hittables hs = sobjs.to_hittables();
    world = sobjs;
    box1.g_free();
    box2.g_free();
    sg.g_free();
  }
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
