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
      mkSolidColorParam(Color(15.0f, 15.0f, 15.0f));

  const MaterialParam red_param = mkLambertParam(red_solid);
  //
  // const float fzz = 0.1f;
  const MaterialParam green_param =
      mkLambertParam(green_solid);
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
  int group_id = 0;
  Primitive ps[] = {
      Primitive(green_param, green_wall, pcount0, group_id),
      Primitive(red_param, red_wall, pcount1, group_id),
      Primitive(light_param, light_wall, pcount2, group_id),
      Primitive(white_param, white_wall, pcount3, group_id),
      Primitive(white_param, white_wall2, pcount4,
                group_id),
      Primitive(blue_param, blue_wall, pcount5, group_id)};
  const MaterialParam mpp;
  const float g_dens = 0.0f;
  GroupParam sg(ps, prim_count, group_id, BOX, g_dens, mpp);

  // a glass sphere
  const TextureParam tp;
  const TextureParam tp2 = mkNoiseParam(2.0f);
  MaterialParam lamb = mkLambertParam(tp2);
  HittableParam hsp1 = mkSphereHittable(
      Point3(190.0f, 350.0f, 290.0f), 90.0f);
  Primitive glass_sphere(lamb, hsp1, 0, 1);
  Primitive ps1[] = {glass_sphere};
  GroupParam sg1(ps1, 1, 1, NONE_GRP, g_dens, mpp);

  // second sphere
  const TextureParam tp3 =
      mkSolidColorParam(Color(1.0f, 0.7f, 0.4f));
  MaterialParam lamb2 = mkLambertParam(tp3);
  MaterialParam mpar(tp3, ISOTROPIC, 0.00f);
  HittableParam hsp2 = mkSphereHittable(
      Point3(370.0f, 95.0f, 265.0f), 90.0f);
  Primitive glass_sphere2(lamb2, hsp2, 0, 2);
  Primitive ps2[] = {glass_sphere2};
  GroupParam sg2(ps2, 1, 2, CONSTANT_MEDIUM, 0.01f, mpar);

  //
  GroupParam *sgs = new GroupParam[3];
  sgs[0] = sg;
  // sgs[2] = box2;
  sgs[1] = sg1;
  sgs[2] = sg2;
  SceneObjects sobjs(sgs, 3);
  // sg.g_free();
  // sg1.g_free();
  // sg2.g_free();
  // smoke.g_free();
  return sobjs;
}

__global__ void make_cornell_box_k(SceneObjects world,
                                   curandState *loc) {
  const TextureParam red_solid =
      mkSolidColorParam(Color(.65, .05, .05));
  const TextureParam green_solid =
      mkSolidColorParam(Color(.12, .75, .15));
  const TextureParam blue_solid =
      mkSolidColorParam(Color(.05, .05, .65));
  const TextureParam white_solid =
      mkSolidColorParam(Color(.75, .75, .75));
  const TextureParam light_solid =
      mkSolidColorParam(Color(15.0f, 15.0f, 15.0f));

  const MaterialParam red_param = mkLambertParam(red_solid);
  //
  // const float fzz = 0.1f;
  const MaterialParam green_param =
      mkLambertParam(green_solid);
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
  int group_id = 0;
  Primitive ps[] = {
      Primitive(green_param, green_wall, pcount0, group_id),
      Primitive(red_param, red_wall, pcount1, group_id),
      Primitive(light_param, light_wall, pcount2, group_id),
      Primitive(white_param, white_wall, pcount3, group_id),
      Primitive(white_param, white_wall2, pcount4,
                group_id),
      Primitive(blue_param, blue_wall, pcount5, group_id)};
  const MaterialParam mpp;
  const float g_dens = 0.0f;
  GroupParam sg(ps, prim_count, group_id, BOX, g_dens, mpp);

  // a glass sphere
  const TextureParam tp;
  const TextureParam tp2 = mkNoiseParam(2.0f);
  MaterialParam lamb = mkLambertParam(tp2);
  HittableParam hsp1 = mkSphereHittable(
      Point3(190.0f, 350.0f, 290.0f), 90.0f);
  Primitive glass_sphere(lamb, hsp1, 0, 1);
  Primitive ps1[] = {glass_sphere};
  GroupParam sg1(ps1, 1, 1, NONE_GRP, g_dens, mpp);

  // second sphere
  const TextureParam tp3 =
      mkSolidColorParam(Color(1.0f, 0.7f, 0.4f));
  MaterialParam lamb2 = mkLambertParam(tp3);
  MaterialParam mpar(tp3, ISOTROPIC, 0.00f);
  HittableParam hsp2 = mkSphereHittable(
      Point3(370.0f, 95.0f, 265.0f), 90.0f);
  Primitive glass_sphere2(lamb2, hsp2, 0, 2);
  Primitive ps2[] = {glass_sphere2};
  GroupParam sg2(ps2, 1, 2, CONSTANT_MEDIUM, 0.01f, mpar);

  //
  GroupParam *sgs = new GroupParam[3];
  sgs[0] = sg;
  // sgs[2] = box2;
  sgs[1] = sg1;
  sgs[2] = sg2;
  SceneObjects sobjs(sgs, 3, loc);
  // sg.g_free();
  // sg1.g_free();
  // sg2.g_free();
  // smoke.g_free();
  world = sobjs;
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
